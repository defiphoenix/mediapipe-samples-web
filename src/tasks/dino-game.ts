/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Dino Game: a minimal "dino run" game driven by the MediaPipe DecisionMaker
 * task (@mediapipe/tasks-decision).
 *
 * Supports two input modalities:
 * - Scene Image (Vision): captures the full 800×220 game scene (dino on the
 *   left, shaded near action zone, and all near & far obstacles on the track—
 *   with no text labels) and compares it against encoded reference full-scene
 *   prototypes (`options[].imageBytes`) in DecisionMaker so the model can
 *   distinguish clear track vs. far obstacles (`wait`) vs. near ground/flying
 *   obstacles (`jump` / `duck`), including multi-obstacle scenes.
 * - Text State: describes what is ahead in words ("Low flying obstacle ahead:
 *   pterodactyl.") and asks a Choice question: jump, duck, or wait.
 */

import template from '../templates/dino-game.html?raw';
import { InferenceTimer } from '../components/inference-timer';
import { DECISION_MODELS, decisionRuntime } from '../components/decision-runtime';
import { mountDecisionModelPanel, type DecisionModelPanelHandle } from '../components/decision-model-panel';

// ---------------------------------------------------------------------------
// Constants (all units are pixels and ticks; one tick = 1/60 s at 1x speed)
// ---------------------------------------------------------------------------

const WIDTH = 800;
const HEIGHT = 220;
const GROUND_Y = 180;

/** Width of the shaded near lookahead action zone in front of the dino. */
const FULL_SCENE_ZONE_W = 200;

const DINO_X = 60;
const DINO_W = 24;
const DINO_H = 28;
const DUCK_W = 32;
const DUCK_H = 16;

const GRAVITY = 0.6;
const JUMP_VELOCITY = 11;
const AIR_TIME = (2 * JUMP_VELOCITY) / GRAVITY; // ticks spent in the air
const APEX_TIME = JUMP_VELOCITY / GRAVITY; // ticks to reach the top of a jump

const START_SPEED = 5;
const MAX_SPEED = 10;
const ACCELERATION = 0.001;

const TICK_MS = 1000 / 60;

export type InputMode = 'image' | 'text';

/**
 * Time the model gets to answer: an obstacle comes into the near action zone
 * this many ticks before the dino must start its jump (16 ticks = ~267 ms for
 * WebGPU vision, 6 ticks = 100 ms for text).
 */
const DECISION_BUDGET_TICKS_IMAGE = 16;
const DECISION_BUDGET_TICKS_TEXT = 6;
const FAR_MIN_MARGIN_TICKS = 12;

function budgetTicksForMode(mode: InputMode): number {
  return mode === 'image' ? DECISION_BUDGET_TICKS_IMAGE : DECISION_BUDGET_TICKS_TEXT;
}

/** The model is asked about a near obstacle once it is this close (grows with speed). */
function lookDistance(speed: number, mode: InputMode = 'image'): number {
  return speed * (APEX_TIME + budgetTicksForMode(mode));
}
/** Start ducking this many ticks before a flying obstacle reaches the dino. */
const DUCK_LEAD_TICKS = 6;

/** Flying obstacles: between ducking height and standing head height. */
const BIRD_BOTTOM = 22;

/**
 * The Choice question asked about the scene text. This wording was picked
 * from a sweep with laya_s256 (13/13 correct on the game's sentences).
 */
const DINO_TEXT_QUESTION = {
  instructions: 'What should the running dino do?',
  options: [
    { label: 'jump', description: 'an obstacle on the ground' },
    { label: 'duck', description: 'an obstacle flying low' },
    { label: 'wait', description: 'no obstacle, the path is clear' },
  ],
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Action = 'JUMP' | 'DUCK' | 'WAIT';
const ACTIONS: Action[] = ['JUMP', 'DUCK', 'WAIT'];

interface ObstacleKind {
  name: string;
  flying: boolean;
  width: number;
  height: number;
  /** Ground obstacles: number of cacti drawn side by side. */
  count?: number;
}

const OBSTACLE_KINDS: ObstacleKind[] = [
  { name: 'small cactus', flying: false, width: 14, height: 26 },
  { name: 'cactus', flying: false, width: 18, height: 36 },
  { name: 'tall cactus', flying: false, width: 18, height: 48 },
  { name: 'two cacti', flying: false, width: 34, height: 32, count: 2 },
  { name: 'three cacti', flying: false, width: 48, height: 28, count: 3 },
  { name: 'bird', flying: true, width: 26, height: 16 },
  { name: 'pterodactyl', flying: true, width: 26, height: 16 },
];

function findObstacleKind(name: string): ObstacleKind {
  return OBSTACLE_KINDS.find((k) => k.name === name) ?? OBSTACLE_KINDS[1];
}

interface PrototypeObstacleSpec {
  kindName: string;
  x: number;
}

interface ScenePrototypeSpec {
  id: string;
  action: Action;
  shortTitle: string;
  obstacles: PrototypeObstacleSpec[];
}

/**
 * Reference full-game scene prototypes (800×220) encoded as `options[].imageBytes`.
 * Covers:
 * - Clear track (`WAIT`)
 * - Far ground or flying obstacle outside the near action zone (`WAIT`)
 * - Near ground obstacle inside the action zone, alone or with a far obstacle (`JUMP`)
 * - Near low-flying obstacle inside the action zone, alone or with a far obstacle (`DUCK`)
 */
const SCENE_PROTOTYPE_SPECS: ScenePrototypeSpec[] = [
  { id: 'wait_clear', action: 'WAIT', shortTitle: 'clear track', obstacles: [] },
  {
    id: 'wait_far_cactus_mid',
    action: 'WAIT',
    shortTitle: 'far cactus (mid)',
    obstacles: [{ kindName: 'cactus', x: 440 }],
  },
  {
    id: 'wait_far_cactus_right',
    action: 'WAIT',
    shortTitle: 'far cactus (right)',
    obstacles: [{ kindName: 'tall cactus', x: 620 }],
  },
  {
    id: 'wait_far_bird_mid',
    action: 'WAIT',
    shortTitle: 'far bird (mid)',
    obstacles: [{ kindName: 'bird', x: 440 }],
  },
  {
    id: 'wait_far_bird_right',
    action: 'WAIT',
    shortTitle: 'far bird (right)',
    obstacles: [{ kindName: 'pterodactyl', x: 620 }],
  },
  {
    id: 'jump_near_cactus_entry',
    action: 'JUMP',
    shortTitle: 'near cactus',
    obstacles: [{ kindName: 'cactus', x: 235 }],
  },
  {
    id: 'jump_near_cactus_mid',
    action: 'JUMP',
    shortTitle: 'near cactus (close)',
    obstacles: [{ kindName: 'cactus', x: 155 }],
  },
  {
    id: 'jump_near_tall_cactus',
    action: 'JUMP',
    shortTitle: 'near tall cactus',
    obstacles: [{ kindName: 'tall cactus', x: 230 }],
  },
  {
    id: 'jump_near_two_cacti',
    action: 'JUMP',
    shortTitle: 'near cactus cluster',
    obstacles: [{ kindName: 'two cacti', x: 220 }],
  },
  {
    id: 'jump_near_cactus_far_cactus',
    action: 'JUMP',
    shortTitle: 'near cactus + far cactus',
    obstacles: [
      { kindName: 'cactus', x: 230 },
      { kindName: 'tall cactus', x: 560 },
    ],
  },
  {
    id: 'jump_near_cactus_far_bird',
    action: 'JUMP',
    shortTitle: 'near cactus + far bird',
    obstacles: [
      { kindName: 'cactus', x: 230 },
      { kindName: 'bird', x: 560 },
    ],
  },
  {
    id: 'duck_near_bird_entry',
    action: 'DUCK',
    shortTitle: 'near bird',
    obstacles: [{ kindName: 'bird', x: 235 }],
  },
  {
    id: 'duck_near_bird_mid',
    action: 'DUCK',
    shortTitle: 'near bird (close)',
    obstacles: [{ kindName: 'bird', x: 155 }],
  },
  {
    id: 'duck_near_bird_far_cactus',
    action: 'DUCK',
    shortTitle: 'near bird + far cactus',
    obstacles: [
      { kindName: 'bird', x: 230 },
      { kindName: 'cactus', x: 560 },
    ],
  },
  {
    id: 'duck_near_bird_far_bird',
    action: 'DUCK',
    shortTitle: 'near bird + far bird',
    obstacles: [
      { kindName: 'bird', x: 230 },
      { kindName: 'bird', x: 560 },
    ],
  },
];

interface Obstacle {
  id: number;
  kind: ObstacleKind;
  x: number;
  passed: boolean;
}

interface World {
  tick: number;
  speed: number;
  dinoY: number; // height above the ground
  dinoVy: number;
  ducking: boolean;
  obstacles: Obstacle[];
  nextId: number;
  nextSpawnIn: number;
  scroll: number; // total distance travelled, for ground animation
  cleared: number;
  gameOver: boolean;
}

/** What the agent can "see" each tick. */
interface Observation {
  onGround: boolean;
  speed: number;
  /** The obstacle inside the near lookahead action zone, if any. */
  target: Obstacle | null;
  distance: number | null; // gap between the dino and `target`
  /** In image mode, a far obstacle on the track outside the near action zone. */
  farTarget: Obstacle | null;
  farDistance: number | null;
  /** Key identifying the current decision scenario ('clear', 'far:<id>', 'near:<id>'). */
  sceneKey: string;
}

interface Decision {
  action: Action;
  reason: string;
  /** Near obstacle id this decision will act on (null = clear or far-only). */
  target: number | null;
  /** Scene key this decision was computed for ('clear', 'far:<id>', 'near:<id>'). */
  sceneKey: string;
  /** Prototype ID matched in image mode. */
  matchedProtoId?: string;
  /** Text sent to the model, if it was called in text mode. */
  prompt?: string;
  probabilities?: Record<Action, number>;
  inferenceTime?: number;
  /** Placeholder while an async model call is in flight. */
  thinking?: boolean;
}

// ---------------------------------------------------------------------------
// Scene text and timing
// ---------------------------------------------------------------------------

/** The text the model reads: what's ahead, in words only. */
function describe(obs: Observation): string {
  if (!obs.target) return 'The path ahead is clear.';
  const { name, flying } = obs.target.kind;
  return flying ? `Low flying obstacle ahead: ${name}.` : `Ground obstacle ahead: ${name}.`;
}

/** Distance at which a jump puts the top of the arc above the middle of the obstacle. */
function jumpDistance(obs: Observation): number {
  return obs.speed * APEX_TIME - (obs.target!.kind.width + DINO_W) / 2;
}

/** The moves to perform this tick for the current decision (timing only). */
function controls(obs: Observation, decision: Decision): { jump: boolean; duck: boolean } {
  const none = { jump: false, duck: false };
  if (!obs.target || decision.target !== obs.target.id || decision.thinking || !obs.onGround) return none;
  if (decision.action === 'JUMP') return { jump: obs.distance! <= jumpDistance(obs), duck: false };
  if (decision.action === 'DUCK') return { jump: false, duck: obs.distance! <= obs.speed * DUCK_LEAD_TICKS };
  return none;
}

// ---------------------------------------------------------------------------
// World simulation
// ---------------------------------------------------------------------------

/** Small seeded PRNG so a run is reproducible while stepping. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createWorld(): World {
  return {
    tick: 0,
    speed: START_SPEED,
    dinoY: 0,
    dinoVy: 0,
    ducking: false,
    obstacles: [],
    nextId: 1,
    nextSpawnIn: 40,
    scroll: 0,
    cleared: 0,
    gameOver: false,
  };
}

function observe(world: World, mode: InputMode = 'image'): Observation {
  const unpassed = world.obstacles.filter((o) => !o.passed && o.x < WIDTH - 8);
  const next = unpassed[0] ?? null;
  const distance = next ? Math.max(0, next.x - (DINO_X + DINO_W)) : null;
  const look = lookDistance(world.speed, mode);
  const inNearZone = Boolean(next && distance !== null && distance <= look);

  if (mode === 'text') {
    const target = inNearZone ? next : null;
    return {
      onGround: world.dinoY === 0,
      speed: world.speed,
      target,
      distance: target ? distance : null,
      farTarget: null,
      farDistance: null,
      sceneKey: target ? `near:${target.id}` : 'clear',
    };
  }

  if (inNearZone && next) {
    const second = unpassed.find((o) => o.id !== next.id && o.x > DINO_X + DINO_W + look) ?? null;
    const secondDist = second ? Math.max(0, second.x - (DINO_X + DINO_W)) : null;
    return {
      onGround: world.dinoY === 0,
      speed: world.speed,
      target: next,
      distance,
      farTarget: second,
      farDistance: secondDist,
      sceneKey: `near:${next.id}`,
    };
  }

  const farCandidate = next && next.x <= WIDTH - 60 ? next : null;
  const farDist = farCandidate ? Math.max(0, farCandidate.x - (DINO_X + DINO_W)) : null;
  return {
    onGround: world.dinoY === 0,
    speed: world.speed,
    target: null,
    distance: null,
    farTarget: farCandidate,
    farDistance: farDist,
    sceneKey: farCandidate ? `far:${farCandidate.id}` : 'clear',
  };
}

/** Advance the world by one tick. */
function stepWorld(world: World, input: { jump: boolean; duck: boolean }, rand: () => number) {
  world.tick++;
  world.scroll += world.speed;

  // Dino physics.
  if (input.jump && world.dinoY === 0) world.dinoVy = JUMP_VELOCITY;
  world.dinoVy -= GRAVITY;
  world.dinoY = Math.max(0, world.dinoY + world.dinoVy);
  if (world.dinoY === 0) world.dinoVy = 0;
  world.ducking = input.duck && world.dinoY === 0;

  // Move obstacles; count the ones the dino got past.
  for (const o of world.obstacles) {
    o.x -= world.speed;
    if (!o.passed && o.x + o.kind.width < DINO_X) {
      o.passed = true;
      world.cleared++;
    }
  }
  world.obstacles = world.obstacles.filter((o) => o.x + o.kind.width > -10);

  // Spawn obstacles with a gap that allows recovery while frequently keeping
  // both a near obstacle and a far obstacle visible on the 800px track.
  world.nextSpawnIn -= world.speed;
  if (world.nextSpawnIn <= 0) {
    const kind = OBSTACLE_KINDS[Math.floor(rand() * OBSTACLE_KINDS.length)];
    world.obstacles.push({ id: world.nextId++, kind, x: WIDTH, passed: false });
    const minGap = Math.max(330, world.speed * AIR_TIME * 1.55);
    world.nextSpawnIn = minGap + rand() * 160;
  }

  world.speed = Math.min(MAX_SPEED, world.speed + ACCELERATION);

  // Collision (axis-aligned boxes, measured upward from the ground).
  const dinoW = world.ducking ? DUCK_W : DINO_W;
  const dinoH = world.ducking ? DUCK_H : DINO_H;
  for (const o of world.obstacles) {
    const bottom = o.kind.flying ? BIRD_BOTTOM : 0;
    const overlapX = DINO_X + dinoW > o.x + 2 && DINO_X < o.x + o.kind.width - 2;
    const overlapY = world.dinoY < bottom + o.kind.height - 2 && world.dinoY + dinoH > bottom + 2;
    if (overlapX && overlapY) {
      world.gameOver = true;
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Sprites
// ---------------------------------------------------------------------------

const DINO_COLOR = '#535353';
const OBSTACLE_COLOR = '#202124';
const PIXEL = 2; // each sprite pixel is 2x2 canvas px

/** Pixel-art T-rex body (rows 0-11), facing right. '.' = empty, 'o' = eye. */
const DINO_BODY = [
  '......XXXXX.',
  '.....XXoXXXX',
  '.....XXXXXXX',
  '.....XXXXXXX',
  '.....XXXX...',
  '.....XXXXXX.',
  'X...XXXXX...',
  'XX.XXXXXXXX.',
  'XXXXXXXXX.X.',
  '.XXXXXXXX...',
  '..XXXXXXX...',
  '...XXXXX....',
];

/** Leg rows (12-13): standing/jumping plus two running frames. */
const DINO_LEGS = {
  stand: ['...XX.XX....', '...XX.XX....'],
  run1: ['...XX.XX....', '...XX.......'],
  run2: ['...XX.XX....', '......XX....'],
};

/** Ducking T-rex (16x6 body + 2 leg rows = DUCK_W x DUCK_H). */
const DINO_DUCK_BODY = [
  '..........XXXXX.',
  'X.XXXXXXXXXoXXXX',
  'XXXXXXXXXXXXXXXX',
  '.XXXXXXXXXXXXXXX',
  '..XXXXXXXXXXX...',
  '...XXXXXXXX.X...',
];
const DINO_DUCK_LEGS = {
  run1: ['....XX..XX......', '....X....XX.....'],
  run2: ['....XX..XX......', '.....XX..X......'],
};

/** Pterodactyl-style bird facing left, wings up / down (13x8 = 26x16 px). */
const BIRD_FRAMES = [
  [
    '.....X.......',
    '.....XX......',
    '.XX..XXX.....',
    'XXXXXXXXXXXXX',
    '....XXXXXXX..',
    '.....XXX.....',
    '.............',
    '.............',
  ],
  [
    '.............',
    '.............',
    '.XX..........',
    'XXXXXXXXXXXXX',
    '....XXXXXXX..',
    '.....XXX.....',
    '.....XX......',
    '.....X.......',
  ],
];

function drawSprite(ctx: CanvasRenderingContext2D, rows: string[], x: number, y: number, crashed = false) {
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length; c++) {
      const ch = crashed && row[c] === 'o' ? 'x' : row[c];
      if (ch === '.') continue;
      ctx.fillStyle = ch === 'o' ? '#fff' : ch === 'x' ? '#d93025' : DINO_COLOR;
      ctx.fillRect(x + c * PIXEL, y + r * PIXEL, PIXEL, PIXEL);
    }
  });
}

/** A black pixel-art cactus (trunk + two arms) at x with the given size. */
function drawCactus(ctx: CanvasRenderingContext2D, x: number, width: number, height: number, groundY = GROUND_Y) {
  const cols = Math.max(5, Math.round(width / PIXEL));
  const rows = Math.round(height / PIXEL);
  const top = groundY - rows * PIXEL;
  const px = (c: number, r: number, w: number, h: number) =>
    ctx.fillRect(x + c * PIXEL, top + r * PIXEL, w * PIXEL, h * PIXEL);

  ctx.fillStyle = OBSTACLE_COLOR;

  // Trunk with a rounded top.
  const trunkW = Math.max(3, Math.round(cols * 0.34));
  const trunkL = Math.floor((cols - trunkW) / 2);
  px(trunkL + 1, 0, trunkW - 2, 1);
  px(trunkL, 1, trunkW, rows - 1);

  // Arms: an elbow on each side, the left one lower than the right.
  const side = trunkL;
  if (side < 2) return;
  const armW = Math.min(2, side - 1);
  const arm = (col: number, armTop: number, armBottom: number, joinFrom: number, joinTo: number) => {
    px(col, armTop + 1, armW, armBottom - armTop); // upright part
    if (armW > 1)
      px(col + 1, armTop, armW - 1, 1); // rounded tip
    else px(col, armTop, 1, 1);
    px(joinFrom, armBottom - armW + 1, joinTo - joinFrom, armW); // joint to trunk
  };
  arm(0, Math.round(rows * 0.3), Math.round(rows * 0.62), armW, trunkL);
  arm(cols - armW, Math.round(rows * 0.18), Math.round(rows * 0.48), trunkL + trunkW, cols - armW);
}

function drawObstacle(
  ctx: CanvasRenderingContext2D,
  o: { kind: ObstacleKind; x: number },
  tick: number,
  groundY = GROUND_Y,
  birdBottom = BIRD_BOTTOM,
  overrideX?: number
) {
  const k = o.kind;
  const drawX = overrideX ?? o.x;
  if (k.flying) {
    const frame = BIRD_FRAMES[Math.floor(tick / 10) % 2];
    ctx.save();
    // Same sprite helper as the dino, but in the obstacle color.
    frame.forEach((row, r) => {
      for (let c = 0; c < row.length; c++) {
        if (row[c] === '.') continue;
        ctx.fillStyle = OBSTACLE_COLOR;
        ctx.fillRect(drawX + c * PIXEL, groundY - birdBottom - k.height + r * PIXEL, PIXEL, PIXEL);
      }
    });
    ctx.restore();
    return;
  }
  const n = k.count ?? 1;
  const each = Math.floor(k.width / n);
  for (let i = 0; i < n; i++) {
    // Vary heights a little within a group.
    const h = k.height - (i % 2) * 6;
    drawCactus(ctx, drawX + i * each, each, h, groundY);
  }
}

/**
 * Renders the full 800×220 game scene (dino on the left, shaded near action
 * zone in front of the dino, ground line across the full width, and all
 * obstacles on the track—with no text overlays or HUD chrome).
 */
function renderFullSceneCanvas(
  canvas: HTMLCanvasElement,
  obstacles: Array<{ kind: ObstacleKind; x: number }>,
  zoneW = FULL_SCENE_ZONE_W
) {
  const sctx = canvas.getContext('2d')!;
  sctx.fillStyle = '#ffffff';
  sctx.fillRect(0, 0, WIDTH, HEIGHT);

  // Shaded near lookahead action zone in front of the dino.
  const zoneX = DINO_X + DINO_W;
  sctx.fillStyle = '#c8eaee';
  sctx.fillRect(zoneX, 20, zoneW, GROUND_Y - 20);
  sctx.fillStyle = '#007f8b';
  sctx.fillRect(zoneX + zoneW, 20, 2, GROUND_Y - 20);

  // Clean horizontal ground line across the full 800px scene.
  sctx.fillStyle = DINO_COLOR;
  sctx.fillRect(0, GROUND_Y, WIDTH, 2);

  // Dino on the left.
  drawSprite(sctx, [...DINO_BODY, ...DINO_LEGS.stand], DINO_X, GROUND_Y - DINO_H);

  // All obstacles on the track (both near inside the action zone and far on the right).
  for (const o of obstacles) {
    if (o.x + o.kind.width < 0 || o.x >= WIDTH) continue;
    drawObstacle(sctx, o, 0, GROUND_Y, BIRD_BOTTOM);
  }
}

/**
 * Renders the live world onto the 800×220 full-scene canvas, normalizing
 * horizontal distance relative to the current speed's lookDistance so the
 * near action zone boundary always aligns with FULL_SCENE_ZONE_W.
 */
function renderFullSceneFromWorld(canvas: HTMLCanvasElement, w: World) {
  const zoneX = DINO_X + DINO_W;
  const baseLook = lookDistance(START_SPEED, 'image');
  const currentLook = lookDistance(w.speed, 'image');
  const scale = currentLook > 0 ? baseLook / currentLook : 1;

  const mappedObstacles: Array<{ kind: ObstacleKind; x: number }> = [];
  for (const o of w.obstacles) {
    if (o.passed || o.x >= WIDTH) continue;
    const relX = Math.max(0, o.x - zoneX);
    const drawX = Math.round(zoneX + relX * scale);
    if (drawX < WIDTH - 4) {
      mappedObstacles.push({ kind: o.kind, x: drawX });
    }
  }
  renderFullSceneCanvas(canvas, mappedObstacles, FULL_SCENE_ZONE_W);
}

async function canvasToJpegBytes(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.95));
  if (!blob) return new Uint8Array(0);
  return new Uint8Array(await blob.arrayBuffer());
}

/** Ground line with scrolling pebbles. */
function drawGround(ctx: CanvasRenderingContext2D, scroll: number) {
  ctx.fillStyle = DINO_COLOR;
  ctx.fillRect(0, GROUND_Y, WIDTH, 1);
  const spacing = 37;
  const offset = scroll % spacing;
  for (let i = 0; i * spacing - offset < WIDTH + spacing; i++) {
    const x = i * spacing - offset;
    const seed = (i + Math.floor(scroll / spacing)) * 9301;
    const dy = 4 + (seed % 7);
    const w = 2 + (seed % 3);
    ctx.fillRect(x, GROUND_Y + dy, w, 1);
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

class DinoGameTask {
  private world = createWorld();
  private rand = mulberry32(Date.now());
  /** The model's current decision about what's ahead; performed by the game when timing is right. */
  private plan: Decision = {
    action: 'WAIT',
    reason: 'Press Play or Next Step.',
    target: null,
    sceneKey: 'init',
  };
  private modelCalls = 0;
  private inputMode: InputMode = 'image';

  private running = false;
  private busy = false;
  /** Shared "Inference Time" badge + history graph, same as the other tasks. */
  private inferenceTimer = new InferenceTimer({ mode: 'single', labelA: 'Decision' });
  /** Bumped on reset/model change so stale async answers are ignored. */
  private epoch = 0;
  private inFlight = false;
  private simSpeed = 1;
  private pauseOnDecision = true;
  private disposers: (() => void)[] = [];
  private modelPanelHandle?: DecisionModelPanelHandle;

  private ctx!: CanvasRenderingContext2D;
  private sceneCanvas!: HTMLCanvasElement;
  private el: Record<string, HTMLElement> = {};
  private protoCards = new Map<string, HTMLElement>();
  private imageQuestionPromise?: Promise<{
    instructions: string;
    temperature: number;
    options: Array<{ label: string; description: string; imageBytes: Uint8Array }>;
  }>;

  constructor(private container: HTMLElement) {}

  init() {
    this.container.innerHTML = template;
    this.container.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));
    this.container.style.setProperty('--dm-width', `${WIDTH}px`);
    this.container.style.setProperty('--dm-aspect', `${WIDTH} / ${HEIGHT}`);
    this.el['dm-question-text'].textContent = DINO_TEXT_QUESTION.instructions;
    this.el['dm-options'].innerHTML = DINO_TEXT_QUESTION.options
      .map((o) => `<li><b>${o.label}</b> — ${o.description}</li>`)
      .join('');

    const canvas = this.el['dm-canvas'] as HTMLCanvasElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    this.ctx = canvas.getContext('2d')!;
    this.ctx.scale(dpr, dpr);

    this.sceneCanvas = this.el['dm-scene-canvas'] as HTMLCanvasElement;
    renderFullSceneFromWorld(this.sceneCanvas, this.world);
    this.mountPrototypeGrid();
    void this.ensureImageQuestion();

    this.el['dm-play'].addEventListener('click', () => this.setRunning(!this.running));
    this.el['dm-step'].addEventListener('click', () => this.step());
    this.el['dm-reset'].addEventListener('click', () => this.reset());
    this.el['dm-speed'].addEventListener('input', (e) => {
      this.simSpeed = parseFloat((e.target as HTMLInputElement).value);
      this.el['dm-speed-value'].textContent = `${this.simSpeed}x`;
    });
    this.el['dm-pause-on-decision'].addEventListener('change', (e) => {
      this.pauseOnDecision = (e.target as HTMLInputElement).checked;
    });
    this.el['dm-mode-image']?.addEventListener('click', () => this.setInputMode('image'));
    this.el['dm-mode-text']?.addEventListener('click', () => this.setInputMode('text'));
    window.addEventListener('keydown', this.onKeyDown);

    // Model controls and state are shared with the Decision Maker page.
    this.modelPanelHandle = mountDecisionModelPanel(this.el['dm-model-panel'], 'vision');
    this.disposers.push(this.modelPanelHandle);
    this.disposers.push(
      decisionRuntime.subscribe((event) => {
        if (event.type === 'loading') {
          // A new model is loading: stop and forget answers from the old one.
          this.setRunning(false);
          this.epoch++;
          this.inFlight = false;
          this.inferenceTimer.resetRollingWindow();
        } else if (event.type === 'ready') {
          this.refreshPlan();
        } else if (event.type === 'status') {
          this.setStatus(event.text);
        } else if (event.type === 'error') {
          this.setStatus(`Error: ${event.error}`);
          this.setRunning(false);
        } else if (event.type === 'state') {
          this.updateButtons();
        }
      })
    );

    this.syncInputModeUI();
    this.render();
    this.updatePanel();
    this.inferenceTimer.mount();

    if (decisionRuntime.ready) {
      if (this.inputMode === 'image' && !this.isLoadedModelVisionCapable()) {
        decisionRuntime.select('embeddinggemma2_text_vision_440m');
        this.modelPanelHandle?.setModality('vision');
        void decisionRuntime.load();
      } else {
        this.refreshPlan();
      }
    } else if (decisionRuntime.loading) {
      this.setStatus(`Loading ${decisionRuntime.loadingLabel}...`);
    } else {
      this.setStatus('Load a vision model to begin');
    }
  }

  /** Renders miniature full-scene prototype cards in the Model Input section. */
  private mountPrototypeGrid() {
    const grid = this.el['dm-proto-grid'];
    if (!grid) return;
    grid.innerHTML = '';
    this.protoCards.clear();

    for (const spec of SCENE_PROTOTYPE_SPECS) {
      const card = document.createElement('div');
      card.className = 'dm-proto-card';
      card.dataset.protoId = spec.id;

      const thumb = document.createElement('canvas');
      thumb.width = WIDTH;
      thumb.height = HEIGHT;
      thumb.className = 'dm-proto-thumb';
      const obstacles = spec.obstacles.map((o) => ({ kind: findObstacleKind(o.kindName), x: o.x }));
      renderFullSceneCanvas(thumb, obstacles, FULL_SCENE_ZONE_W);

      const meta = document.createElement('div');
      meta.className = 'dm-proto-meta';

      const badge = document.createElement('span');
      badge.className = `dm-proto-badge ${spec.action.toLowerCase()}`;
      badge.textContent = spec.action;

      const title = document.createElement('span');
      title.className = 'dm-proto-title';
      title.title = spec.shortTitle;
      title.textContent = spec.shortTitle;

      meta.append(badge, title);
      card.append(thumb, meta);
      grid.appendChild(card);
      this.protoCards.set(spec.id, card);
    }
  }

  /** Builds and caches the multimodal ChoiceQuestion with encoded full-scene prototypes. */
  private ensureImageQuestion() {
    if (!this.imageQuestionPromise) {
      this.imageQuestionPromise = (async () => {
        const offscreen = document.createElement('canvas');
        offscreen.width = WIDTH;
        offscreen.height = HEIGHT;

        const options: Array<{ label: string; description: string; imageBytes: Uint8Array }> = [];
        for (const spec of SCENE_PROTOTYPE_SPECS) {
          const obstacles = spec.obstacles.map((o) => ({ kind: findObstacleKind(o.kindName), x: o.x }));
          renderFullSceneCanvas(offscreen, obstacles, FULL_SCENE_ZONE_W);
          const imageBytes = await canvasToJpegBytes(offscreen);
          options.push({
            label: spec.id,
            description: '',
            imageBytes,
          });
        }
        return {
          instructions: DINO_TEXT_QUESTION.instructions,
          temperature: 0.01,
          options,
        };
      })();
    }
    return this.imageQuestionPromise;
  }

  private isLoadedModelVisionCapable(): boolean {
    if (!decisionRuntime.ready) return false;
    if (decisionRuntime.customModel) return true;
    return Boolean(decisionRuntime.loadedValue && DECISION_MODELS[decisionRuntime.loadedValue]?.supportsVision);
  }

  private setInputMode(mode: InputMode) {
    if (this.inputMode === mode) return;
    this.setRunning(false);
    this.epoch++;
    this.inFlight = false;
    this.inputMode = mode;
    this.modelPanelHandle?.setModality(mode === 'image' ? 'vision' : 'text');
    this.syncInputModeUI();
    this.render();
    this.updatePanel();

    if (mode === 'image' && decisionRuntime.ready && !this.isLoadedModelVisionCapable()) {
      decisionRuntime.select('embeddinggemma2_text_vision_440m');
      this.modelPanelHandle?.setModality('vision');
      void decisionRuntime.load();
      return;
    }
    if (decisionRuntime.ready) {
      void this.refreshPlan();
    }
  }

  private syncInputModeUI() {
    const isImage = this.inputMode === 'image';
    this.el['dm-mode-image']?.classList.toggle('active', isImage);
    this.el['dm-mode-text']?.classList.toggle('active', !isImage);
    if (this.el['dm-help']) {
      this.el['dm-help'].innerHTML = isImage
        ? 'The game captures the <b>full 800×220 game scene</b> (dino on the left, shaded near action zone, and all near &amp; far obstacles on the track—with no text labels) and compares those pixels against encoded reference full-scene prototypes in <code>DecisionMaker</code> to choose: <b>jump</b>, <b>duck</b>, or <b>wait</b>.'
        : 'The model never sees the game. When something new comes into view, it gets a one-line text description and makes a Choice: <b>jump</b>, <b>duck</b>, or <b>wait</b>.';
    }
    if (this.el['dm-input-label']) {
      this.el['dm-input-label'].textContent = isImage
        ? 'Model input (full 800×220 game scene)'
        : 'Model input (scene text description)';
    }
    if (this.el['dm-input-badge']) {
      this.el['dm-input-badge'].textContent = isImage ? 'Vision · { imageBytes }' : 'Text · string';
    }
    if (this.el['dm-scene-preview-wrap']) {
      this.el['dm-scene-preview-wrap'].style.display = isImage ? 'flex' : 'none';
    }
    if (this.el['dm-proto-section']) {
      this.el['dm-proto-section'].style.display = isImage ? 'block' : 'none';
    }
    if (this.el['dm-options']) {
      this.el['dm-options'].style.display = isImage ? 'none' : 'block';
    }
    if (this.el['dm-prompt']) {
      this.el['dm-prompt'].style.display = isImage ? 'none' : 'block';
    }
  }

  cleanup() {
    this.running = false;
    this.epoch++;
    window.removeEventListener('keydown', this.onKeyDown);
    this.inferenceTimer.cleanup();
    // The worker and model stay loaded in decisionRuntime for the next page.
    for (const dispose of this.disposers) dispose();
    this.disposers = [];
  }

  // -------------------------------------------------------------------------
  // Model (shared runtime, runs in a worker)
  // -------------------------------------------------------------------------

  /**
   * True if the scene state changed since the last decision (or a decision is
   * still pending). In Scene Image mode, we evaluate:
   * - 'clear' when no obstacles are on screen,
   * - 'far:<id>' when an obstacle is on the far track (and the dino is on the
   *   ground with plenty of margin before the near action zone), and
   * - 'near:<id>' as soon as an obstacle enters the near lookahead action zone.
   */
  private needsDecision(obs: Observation): boolean {
    if (this.plan.thinking && !this.inFlight) return true;
    if (this.plan.sceneKey === obs.sceneKey) return false;

    if (this.inputMode === 'image' && obs.sceneKey.startsWith('far:')) {
      if (!obs.onGround || obs.farDistance === null) return false;
      const minFarDist =
        lookDistance(obs.speed, this.inputMode) + obs.speed * (DECISION_BUDGET_TICKS_IMAGE + FAR_MIN_MARGIN_TICKS);
      return obs.farDistance >= minFarDist;
    }
    return true;
  }

  /** Asks the model about the scene: one Choice call (either full-scene image or text state). */
  private async modelDecide(obs: Observation): Promise<Decision> {
    const target = obs.target?.id ?? null;
    const sceneKey = obs.sceneKey;
    if (!decisionRuntime.ready) return { action: 'WAIT', reason: 'Model not ready.', target, sceneKey };

    if (this.inputMode === 'image') {
      if (!this.isLoadedModelVisionCapable()) {
        return {
          action: 'WAIT',
          reason: 'Select and load a vision-capable model (440M or 740M).',
          target,
          sceneKey,
        };
      }
      renderFullSceneFromWorld(this.sceneCanvas, this.world);
      const [imageBytes, imageQuestion] = await Promise.all([
        canvasToJpegBytes(this.sceneCanvas),
        this.ensureImageQuestion(),
      ]);
      if (this.el['dm-scene-caption']) {
        this.el['dm-scene-caption'].textContent =
          `800×220 full-scene JPEG · ${(imageBytes.byteLength / 1024).toFixed(1)} KB`;
      }
      this.modelCalls++;
      const msg = await decisionRuntime.ask({ imageBytes }, 'choice', imageQuestion);
      if (msg.type !== 'DECIDE_RESULT') {
        return { action: 'WAIT', reason: `Model error: ${msg.error}`, target, sceneKey };
      }
      const { selectedKey, probabilities } = msg.result as {
        selectedKey: string;
        probabilities: Record<string, number>;
      };
      this.inferenceTimer.record(msg.inferenceTime, 'GPU');

      // Aggregate prototype probabilities by action class using max-pooling so
      // classes with more reference prototypes are not artificially boosted.
      const maxByAction: Record<Action, number> = { JUMP: 0, DUCK: 0, WAIT: 0 };
      let bestSpec = SCENE_PROTOTYPE_SPECS[0];
      let bestRawProb = -1;
      for (const spec of SCENE_PROTOTYPE_SPECS) {
        const p = probabilities?.[spec.id] ?? 0;
        if (p > maxByAction[spec.action]) {
          maxByAction[spec.action] = p;
        }
        if (p > bestRawProb || spec.id === selectedKey) {
          if (p > bestRawProb) {
            bestRawProb = p;
            bestSpec = spec;
          }
        }
      }
      const matchedSpec = SCENE_PROTOTYPE_SPECS.find((s) => s.id === selectedKey) ?? bestSpec;
      const action: Action = matchedSpec.action;
      const sumMax = maxByAction.JUMP + maxByAction.DUCK + maxByAction.WAIT || 1;
      const probs: Record<Action, number> = {
        JUMP: maxByAction.JUMP / sumMax,
        DUCK: maxByAction.DUCK / sumMax,
        WAIT: maxByAction.WAIT / sumMax,
      };

      let sceneDesc = 'clear track';
      if (obs.target && obs.farTarget) {
        sceneDesc = `near ${obs.target.kind.name} in action zone + far ${obs.farTarget.kind.name} ahead`;
      } else if (obs.target) {
        sceneDesc = `near ${obs.target.kind.name} in action zone`;
      } else if (obs.farTarget) {
        sceneDesc = `far ${obs.farTarget.kind.name} outside action zone`;
      }

      return {
        action,
        reason: `Scene: ${sceneDesc} → matched “${matchedSpec.shortTitle}” → ${action.toLowerCase()} (${Math.round(probs[action] * 100)}%).`,
        target,
        sceneKey,
        matchedProtoId: matchedSpec.id,
        probabilities: probs,
        inferenceTime: msg.inferenceTime,
      };
    }

    const prompt = describe(obs);
    this.modelCalls++;
    const msg = await decisionRuntime.ask(prompt, 'choice', DINO_TEXT_QUESTION);
    if (msg.type !== 'DECIDE_RESULT') {
      return { action: 'WAIT', reason: `Model error: ${msg.error}`, target, sceneKey, prompt };
    }

    const { selectedKey, probabilities } = msg.result as {
      selectedKey: string;
      probabilities: Record<string, number>;
    };
    this.inferenceTimer.record(msg.inferenceTime, 'GPU');
    const action = (selectedKey?.toUpperCase() as Action) ?? 'WAIT';
    const probs = Object.fromEntries(ACTIONS.map((a) => [a, probabilities?.[a.toLowerCase()] ?? 0])) as Record<
      Action,
      number
    >;
    const what = obs.target ? `“${obs.target.kind.name}”` : 'a clear path';
    return {
      action: ACTIONS.includes(action) ? action : 'WAIT',
      reason: `Model read ${what} → ${selectedKey} (${Math.round(probs[action] * 100)}%).`,
      target,
      sceneKey,
      prompt,
      probabilities: probs,
      inferenceTime: msg.inferenceTime,
    };
  }

  /**
   * While playing, ask the model in the background so the game keeps its speed.
   * An obstacle enters the near action zone DECISION_BUDGET_TICKS before the
   * dino has to act, so the model answers within that budget.
   */
  private askModelInBackground(obs: Observation) {
    this.inFlight = true;
    const epoch = this.epoch;
    this.modelDecide(obs).then((decision) => {
      this.inFlight = false;
      if (epoch !== this.epoch) return;
      // Ignore answers about a scene state that is no longer current.
      if (decision.sceneKey !== this.plan.sceneKey) return;
      this.plan = decision;
      if (this.running && this.pauseOnDecision && decision.action !== 'WAIT') this.setRunning(false);
      this.render();
      this.updatePanel();
    });
  }

  /** Blocking decision for the current frame (used when paused / stepping). */
  private async refreshPlan() {
    if (this.busy) return;
    this.busy = true;
    try {
      this.plan = await this.modelDecide(observe(this.world, this.inputMode));
    } finally {
      this.busy = false;
    }
    this.render();
    this.updatePanel();
  }

  // -------------------------------------------------------------------------
  // Simulation control
  // -------------------------------------------------------------------------

  private onKeyDown = (e: KeyboardEvent) => {
    const target = e.target as HTMLElement;
    if (
      target &&
      ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName) &&
      target.getAttribute('type') !== 'checkbox'
    )
      return;
    if (e.code === 'Space') {
      e.preventDefault();
      this.setRunning(!this.running);
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      this.step();
    } else if (e.code === 'KeyR') {
      this.reset();
    }
  };

  private canStep() {
    return !this.world.gameOver && decisionRuntime.ready;
  }

  private setRunning(running: boolean) {
    if (running && !this.canStep()) return;
    const wasRunning = this.running;
    this.running = running;
    if (running && !wasRunning) this.runLoop();
    if (!running && wasRunning && this.plan.thinking && !this.inFlight) this.refreshPlan();
    this.updateButtons();
  }

  /** Plays at a steady speed; model answers arrive asynchronously. */
  private async runLoop() {
    while (this.running) {
      const start = performance.now();
      this.tick();
      const wait = TICK_MS / this.simSpeed - (performance.now() - start);
      await new Promise((r) => setTimeout(r, Math.max(0, wait)));
    }
  }

  /** One tick while playing (never waits for the model). */
  private tick() {
    if (this.busy || !this.canStep()) return;
    stepWorld(this.world, controls(observe(this.world, this.inputMode), this.plan), this.rand);

    const obs = observe(this.world, this.inputMode);
    if (this.needsDecision(obs)) {
      const target = obs.target?.id ?? null;
      if (obs.sceneKey !== this.plan.sceneKey) {
        if (this.inputMode === 'image') {
          renderFullSceneFromWorld(this.sceneCanvas, this.world);
        }
        this.plan = {
          action: 'WAIT',
          reason:
            this.inputMode === 'image' ? 'Model is inspecting full scene image…' : 'Model is reading what’s ahead…',
          target,
          sceneKey: obs.sceneKey,
          thinking: true,
        };
      }
      if (!this.inFlight) this.askModelInBackground(obs);
    }

    if (this.world.gameOver) this.setRunning(false);
    this.render();
    this.updatePanel();
  }

  /** One manual step: act, then (if what's ahead changed) wait for the model. */
  private async step() {
    if (this.running || this.busy || !this.canStep()) return;
    this.busy = true;
    try {
      stepWorld(this.world, controls(observe(this.world, this.inputMode), this.plan), this.rand);
      const obs = observe(this.world, this.inputMode);
      if (this.needsDecision(obs)) this.plan = await this.modelDecide(obs);
    } finally {
      this.busy = false;
    }
    this.render();
    this.updatePanel();
  }

  private async reset() {
    this.setRunning(false);
    this.epoch++;
    this.inFlight = false;
    this.world = createWorld();
    this.rand = mulberry32(Date.now());
    this.modelCalls = 0;
    renderFullSceneFromWorld(this.sceneCanvas, this.world);
    await this.refreshPlan();
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private setStatus(text: string) {
    this.el['status-message'].textContent = text;
    this.inferenceTimer.syncStatusVisibility(text);
    this.updateButtons();
  }

  private updateButtons() {
    const play = this.el['dm-play'] as HTMLButtonElement;
    play.querySelector('.material-icons')!.textContent = this.running ? 'pause' : 'play_arrow';
    play.querySelector('.dm-btn-label')!.textContent = this.running ? 'Pause' : 'Play';
    play.disabled = !this.canStep();
    (this.el['dm-step'] as HTMLButtonElement).disabled = this.running || !this.canStep();
  }

  private updatePanel() {
    const d = this.plan;
    this.el['dm-obs-tick'].textContent = `${this.world.tick}`;
    this.el['dm-obs-speed'].textContent = `${this.world.speed.toFixed(2)} px/t`;
    this.el['dm-obs-cleared'].textContent = `${this.world.cleared}`;
    this.el['dm-obs-calls'].textContent = `${this.modelCalls}`;

    const action = this.el['dm-action'];
    const over = this.world.gameOver;
    action.textContent = over ? 'CRASH' : d.thinking ? '…' : d.action;
    action.className = `dm-action ${over ? 'crash' : d.thinking ? '' : d.action.toLowerCase()}`;
    this.el['dm-reason'].textContent = over ? 'Hit an obstacle. Press Reset.' : d.reason;

    for (const a of ACTIONS) {
      const p = d.thinking ? undefined : d.probabilities?.[a];
      (this.el[`dm-prob-${a}`] as HTMLElement).style.width = p === undefined ? '0' : `${p * 100}%`;
      this.el[`dm-prob-value-${a}`].textContent = p === undefined ? '-' : p.toFixed(2);
    }
    if (d.prompt) this.el['dm-prompt'].textContent = d.prompt;

    for (const [id, card] of this.protoCards.entries()) {
      card.classList.toggle('matched', !d.thinking && d.matchedProtoId === id);
    }

    if (decisionRuntime.ready) {
      this.el['status-message'].textContent = over ? 'Game over' : this.running ? 'Running' : 'Paused';
    }
    this.updateButtons();
  }

  private render() {
    const ctx = this.ctx;
    const w = this.world;
    const obs = observe(w, this.inputMode);

    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // Near action zone in front of the dino.
    const look = lookDistance(w.speed, this.inputMode);
    const zoneW =
      this.inputMode === 'image' ? Math.round((FULL_SCENE_ZONE_W / lookDistance(START_SPEED, 'image')) * look) : look;
    ctx.fillStyle = 'rgba(0, 127, 139, 0.10)';
    ctx.fillRect(DINO_X + DINO_W, 20, zoneW, GROUND_Y - 20);
    ctx.strokeStyle = 'rgba(0, 127, 139, 0.65)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(DINO_X + DINO_W + zoneW, 20);
    ctx.lineTo(DINO_X + DINO_W + zoneW, GROUND_Y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#007f8b';
    ctx.font = '11px Roboto, sans-serif';
    const budgetMs = Math.round((budgetTicksForMode(this.inputMode) * TICK_MS) / this.simSpeed);
    ctx.textAlign = 'right';
    ctx.fillText(
      this.inputMode === 'image' ? 'near action zone (jump / duck)' : 'model looks here',
      DINO_X + DINO_W + zoneW - 6,
      32
    );
    ctx.fillText(`${budgetMs} ms to decide`, DINO_X + DINO_W + zoneW - 6, 45);

    if (this.inputMode === 'image') {
      ctx.fillStyle = '#5f6368';
      ctx.textAlign = 'left';
      ctx.fillText('far track (wait until obstacle enters action zone) →', DINO_X + DINO_W + zoneW + 10, 32);
    }
    ctx.textAlign = 'left';

    if (this.sceneCanvas && !this.running) {
      renderFullSceneFromWorld(this.sceneCanvas, w);
    }

    drawGround(ctx, w.scroll);

    // Obstacles: show the model's decision on the active near or far target.
    for (const o of w.obstacles) {
      drawObstacle(ctx, o, w.tick);
      const top = GROUND_Y - (o.kind.flying ? BIRD_BOTTOM : 0) - o.kind.height;
      const isNearTarget = obs.target?.id === o.id && this.plan.sceneKey === `near:${o.id}`;
      const isFarTarget = !obs.target && obs.farTarget?.id === o.id && this.plan.sceneKey === `far:${o.id}`;
      let label = '';
      if (this.inputMode === 'image') {
        if (isNearTarget && !this.plan.thinking) {
          label = `near → ${this.plan.action}`;
        } else if (isFarTarget && !this.plan.thinking) {
          label = `far → ${this.plan.action}`;
        }
      } else {
        label = isNearTarget && !this.plan.thinking ? `${o.kind.name} → ${this.plan.action}` : o.kind.name;
      }
      if (label) {
        ctx.font = `${isNearTarget || isFarTarget ? 'bold ' : ''}11px Roboto, sans-serif`;
        ctx.fillStyle = isNearTarget ? '#007f8b' : '#5f6368';
        ctx.textAlign = 'center';
        ctx.fillText(label, o.x + o.kind.width / 2, top - 6);
        ctx.textAlign = 'left';
      }
    }

    // Dino (legs alternate every 6 ticks while running).
    const frame = Math.floor(w.tick / 6) % 2;
    if (w.ducking && !w.gameOver) {
      const legs = frame ? DINO_DUCK_LEGS.run1 : DINO_DUCK_LEGS.run2;
      drawSprite(ctx, [...DINO_DUCK_BODY, ...legs], DINO_X, GROUND_Y - DUCK_H);
    } else {
      const legs = w.dinoY > 0 || w.gameOver ? DINO_LEGS.stand : frame ? DINO_LEGS.run1 : DINO_LEGS.run2;
      drawSprite(ctx, [...DINO_BODY, ...legs], DINO_X, GROUND_Y - w.dinoY - DINO_H, w.gameOver);
    }

    if (w.gameOver) {
      ctx.fillStyle = DINO_COLOR;
      ctx.font = 'bold 20px Roboto Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('GAME OVER', WIDTH / 2, HEIGHT / 2);
      ctx.textAlign = 'left';
    }
  }
}

let activeTask: DinoGameTask | null = null;

export async function setupDinoGame(container: HTMLElement) {
  activeTask = new DinoGameTask(container);
  activeTask.init();
}

export function cleanupDinoGame() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
