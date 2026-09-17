import { N_REFS, PLANE, RES, type Serve320Bundle } from "../assets/serve320-bundle";
import type { Point } from "./geometry";

/** Cached stratum → row pools so the hot path does not rebuild Maps each frame. */
const codebookPoolCache = new WeakMap<Serve320Bundle, {
  closed: number[];
  mid: number[];
  wide: number[];
  all: number[];
}>();

function codebookPools(bundle: Serve320Bundle) {
  let cached = codebookPoolCache.get(bundle);
  if (cached) return cached;
  const frameToRow = new Map(bundle.codebook.refs.map((entry) => [entry.frame, entry.row]));
  const resolve = (frames: number[]) => frames
    .map((frame) => frameToRow.get(frame))
    .filter((row): row is number => row !== undefined);
  cached = {
    closed: resolve(bundle.codebook.strata.closed),
    mid: resolve(bundle.codebook.strata.mid),
    wide: resolve(bundle.codebook.strata.wide),
    all: Array.from({ length: N_REFS }, (_, index) => index),
  };
  codebookPoolCache.set(bundle, cached);
  return cached;
}

/**
 * Retrieval hysteresis margin.
 *
 * Selection here is per-frame and memoryless, so the exemplar changes on 40% of
 * consecutive frames — 27 unique rows over a reply. That is a real flicker
 * source, and until a codebook-faithful harness existed nothing could see it:
 * the offline harness pinned one reference per frame and switched on 0.02%.
 *
 * 60.8% of those switches are the aperture crossing a pool edge, but the edges
 * are NOT chattering — median |aperture − nearest edge| at a flip is 0.052, and
 * only 22% sit within 0.02 — so a deadband was measured and rejected (it removed
 * 17% of pool changes but only 6.7% of switches; within-pool flips filled the
 * gap). A query EMA was rejected too: it cannot touch the pool by construction.
 *
 * Global hysteresis was the one mechanism whose retrieval distance goes DOWN
 * while switching falls (0.966× at this margin), because the previous row stays
 * eligible even outside the current pool — which also lets it hold a row across
 * an edge, cutting pool-driven switches 1019 → 567. Measured on sealed test:
 *
 *   switching        0.39753 -> 0.25190  (-36.6%), run length 2.47 -> 3.83 frames
 *   temporal_ratio   0.95146  CI [0.884, 1.032]  P(>1.10) 0.0011
 *   silence_flicker  0.96816  CI [0.907, 1.015]  P(>1.10) 0.0
 *   painted_lowpass  0.99666  significantly BETTER
 *   gate rank        whole 0.99060 · win-b8 0.98640 · win-b4 0.99122
 *
 * All 8 gates pass on both renderers. The temporal and flicker gains are
 * directionally consistent everywhere but are NOT significant at 95% on 54
 * clips; what is significant is the appearance improvement and the
 * non-regression. The windowed-seam bar is untouched: worst boundary at
 * bootstrap 4 is identical to the baseline (14.9589) with 0/54 above 20 luma.
 *
 * `margin = 0` is bit-identical to memoryless selection.
 */
export const APPEARANCE_HYSTERESIS_MARGIN = 0.25;

export function appearanceCodebookRow(
  query: ArrayLike<number>,
  aperture: number,
  bundle: Serve320Bundle,
  previousRow = -1,
  margin = 0,
): number {
  const stratum = aperture <= 0.20 ? "closed" : aperture > 0.45 ? "wide" : "mid";
  const pools = codebookPools(bundle);
  let pool = pools[stratum];
  if (pool.length === 0) pool = pools.all;
  const squaredDistance = (row: number): number => {
    let distance = 0;
    for (let k = 0; k < 6; k += 1) {
      const delta = bundle.refGeom6[row * 6 + k] - Number(query[k]);
      distance += delta * delta;
    }
    return distance;
  };
  let best = pool[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const row of pool) {
    const distance = squaredDistance(row);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = row;
    }
  }
  // Hold the previous exemplar unless the pool's best beats it by the margin.
  // Deliberately NOT restricted to the current pool: a within-pool-only variant
  // can only ever hold a worse row, and measured worse on both axes (retrieval
  // cost 1.0104 vs 0.9657, switching 0.336 vs 0.254 at the same margin).
  if (margin > 0 && previousRow >= 0 && previousRow !== best) {
    if (!(bestDistance * (1 + margin) < squaredDistance(previousRow))) return previousRow;
  }
  return best;
}

interface Similarity {
  scale: number;
  rotation: [number, number, number, number];
  translation: Point;
}

export function umeyamaSimilarity(source: Point[], target: Point[]): Similarity {
  const count = source.length;
  let sourceX = 0, sourceY = 0, targetX = 0, targetY = 0;
  for (let i = 0; i < count; i += 1) {
    sourceX += source[i][0]; sourceY += source[i][1];
    targetX += target[i][0]; targetY += target[i][1];
  }
  sourceX /= count; sourceY /= count; targetX /= count; targetY /= count;
  let variance = 0, m00 = 0, m01 = 0, m10 = 0, m11 = 0;
  for (let i = 0; i < count; i += 1) {
    const sx = source[i][0] - sourceX, sy = source[i][1] - sourceY;
    const tx = target[i][0] - targetX, ty = target[i][1] - targetY;
    variance += sx * sx + sy * sy;
    m00 += tx * sx; m01 += tx * sy; m10 += ty * sx; m11 += ty * sy;
  }
  variance /= count; m00 /= count; m01 /= count; m10 /= count; m11 /= count;
  const a = m00 * m00 + m10 * m10;
  const b = m00 * m01 + m10 * m11;
  const c = m01 * m01 + m11 * m11;
  const trace = a + c;
  const determinant = a * c - b * b;
  const discriminant = Math.sqrt(Math.max(0, trace * trace / 4 - determinant));
  const lambda1 = trace / 2 + discriminant;
  const lambda2 = trace / 2 - discriminant;
  let v1x = 1, v1y = 0;
  if (b !== 0) {
    let x = lambda1 - c, y = b;
    let length = Math.hypot(x, y);
    if (length <= 1e-12) { x = b; y = lambda1 - a; length = Math.hypot(x, y); }
    if (length > 1e-12) { v1x = x / length; v1y = y / length; }
  } else if (c > a) { v1x = 0; v1y = 1; }
  const v2x = -v1y, v2y = v1x;
  const singular1 = Math.sqrt(Math.max(lambda1, 0));
  const singular2 = Math.sqrt(Math.max(lambda2, 0));
  let u1x = singular1 > 1e-12 ? (m00 * v1x + m01 * v1y) / singular1 : 1;
  let u1y = singular1 > 1e-12 ? (m10 * v1x + m11 * v1y) / singular1 : 0;
  let length = Math.hypot(u1x, u1y);
  if (length < 1e-6) { u1x = 1; u1y = 0; length = 1; }
  u1x /= length; u1y /= length;
  let u2x = singular2 > 1e-12 ? (m00 * v2x + m01 * v2y) / singular2 : -u1y;
  let u2y = singular2 > 1e-12 ? (m10 * v2x + m11 * v2y) / singular2 : u1x;
  length = Math.hypot(u2x, u2y);
  if (length < 1e-6) { u2x = -u1y; u2y = u1x; length = 1; }
  u2x /= length; u2y /= length;
  const reflection = u1x * u2y - u1y * u2x >= 0 ? 1 : -1;
  u2x *= reflection; u2y *= reflection;
  const r00 = u1x * v1x + u2x * v2x;
  const r01 = u1x * v1y + u2x * v2y;
  const r10 = u1y * v1x + u2y * v2x;
  const r11 = u1y * v1y + u2y * v2y;
  const scale = (singular1 + reflection * singular2) / Math.max(variance, 1e-8);
  return {
    scale,
    rotation: [r00, r01, r10, r11],
    translation: [
      targetX - scale * (r00 * sourceX + r01 * sourceY),
      targetY - scale * (r10 * sourceX + r11 * sourceY),
    ],
  };
}

export function planarBgrToRgb01(planar: Uint8Array): Float32Array {
  const output = new Float32Array(3 * PLANE);
  for (let pixel = 0; pixel < PLANE; pixel += 1) {
    output[pixel] = planar[2 * PLANE + pixel] / 255;
    output[PLANE + pixel] = planar[PLANE + pixel] / 255;
    output[2 * PLANE + pixel] = planar[pixel] / 255;
  }
  return output;
}

export function planarBgrToHwc(planar: Uint8Array): Uint8Array {
  const output = new Uint8Array(3 * PLANE);
  for (let pixel = 0; pixel < PLANE; pixel += 1) {
    output[pixel * 3] = planar[pixel];
    output[pixel * 3 + 1] = planar[PLANE + pixel];
    output[pixel * 3 + 2] = planar[2 * PLANE + pixel];
  }
  return output;
}

export function alignReference(
  reference: Float32Array,
  source: Point[],
  target: Point[],
  destination?: Float32Array,
): Float32Array {
  const similarity = umeyamaSimilarity(source, target);
  const [r00, r01, r10, r11] = similarity.rotation;
  const inverseScale = 1 / Math.max(similarity.scale, 1e-8);
  const output = destination ?? new Float32Array(reference.length);
  if (output.length !== reference.length) {
    throw new Error(`alignReference buffer length ${output.length} != ${reference.length}`);
  }
  for (let y = 0; y < RES; y += 1) {
    for (let x = 0; x < RES; x += 1) {
      const px = x - similarity.translation[0];
      const py = y - similarity.translation[1];
      const rx = Math.min(RES - 1, Math.max(0, (r00 * px + r10 * py) * inverseScale));
      const ry = Math.min(RES - 1, Math.max(0, (r01 * px + r11 * py) * inverseScale));
      const x0 = Math.floor(rx), y0 = Math.floor(ry);
      const x1 = Math.min(x0 + 1, RES - 1), y1 = Math.min(y0 + 1, RES - 1);
      const fx = rx - x0, fy = ry - y0;
      const i00 = y0 * RES + x0, i01 = y0 * RES + x1;
      const i10 = y1 * RES + x0, i11 = y1 * RES + x1;
      const pixel = y * RES + x;
      for (let channel = 0; channel < 3; channel += 1) {
        const base = channel * PLANE;
        output[base + pixel] = reference[base + i00] * (1 - fx) * (1 - fy)
          + reference[base + i01] * fx * (1 - fy)
          + reference[base + i10] * (1 - fx) * fy
          + reference[base + i11] * fx * fy;
      }
    }
  }
  return output;
}
