import type { Serve320Pca } from "../assets/serve320-bundle";

export type Point = [number, number];
export type Anchor = [number, number, number, number];

export const R2B_TEMPLATE_YOFF = new Float32Array([
  -5.809554, -6.667094, -7.0917296, -7.0300374, -7.353298, -8.109091,
  -7.319757, 4.5667334, 14.743867, 19.228, 16.160421, 4.6456633,
  -4.4976387, -0.66066223, 1.155612, -1.8853236, -6.0676365, -0.9888773,
  2.4905596, 0.48984563,
]);

export function reconstructGeometry(
  scores: ArrayLike<number>,
  pca: Serve320Pca,
  destination?: Float32Array,
): Float32Array {
  const result = destination ?? new Float32Array(40);
  for (let j = 0; j < 40; j += 1) {
    let value = pca.geometry_mean[j];
    for (let k = 0; k < 6; k += 1) {
      value += Number(scores[k]) * pca.score_std[k] * pca.components[k][j];
    }
    result[j] = value;
  }
  return result;
}

export function apertureFromGeometry(geometry: ArrayLike<number>): number {
  const dx = Number(geometry[12]) - Number(geometry[0]);
  const dy = Number(geometry[13]) - Number(geometry[1]);
  const width = Math.hypot(dx, dy);
  if (width <= 1e-6) return 0;
  const gaps = [
    Math.max(0, Number(geometry[39]) - Number(geometry[27])),
    Math.max(0, Number(geometry[37]) - Number(geometry[29])),
    Math.max(0, Number(geometry[35]) - Number(geometry[31])),
  ].sort((a, b) => a - b);
  return Math.min(1, Math.max(0, gaps[1] / width / 0.35));
}

export function aperture(scores: ArrayLike<number>, pca: Serve320Pca): number {
  return apertureFromGeometry(reconstructGeometry(scores, pca));
}

export function decodePointsFromGeometry(
  geometry: ArrayLike<number>,
  anchor: Anchor,
  destination?: Point[],
): Point[] {
  const [midX, midY, width, angle] = anchor;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const points = destination ?? Array.from({ length: 20 }, () => [0, 0] as Point);
  for (let i = 0; i < 20; i += 1) {
    const x = Number(geometry[i * 2]);
    const y = Number(geometry[i * 2 + 1]);
    const point = points[i] ?? (points[i] = [0, 0]);
    point[0] = midX + width * (cosine * x - sine * y);
    point[1] = midY + width * (sine * x + cosine * y);
  }
  return points;
}

export function decodePoints(
  scores: ArrayLike<number>,
  anchor: Anchor,
  pca: Serve320Pca,
  geometryScratch?: Float32Array,
  pointsScratch?: Point[],
): Point[] {
  const geometry = reconstructGeometry(scores, pca, geometryScratch);
  return decodePointsFromGeometry(geometry, anchor, pointsScratch);
}

/**
 * Closure gate.
 *
 * The threshold is a property of the HEAD, not a universal constant, and it
 * must move with the model. 0.70 belonged to the wave3 head. The R0-ctrl
 * EMA@2000 head shipped alongside it is recall-lean at 0.70 — measured on 4103
 * rows in this exact sliding 30/8 regime, precision 0.7150 against wave3's
 * 0.7466, i.e. 46 extra false closures. Teeth-showing seals are the owner's
 * most-filed defect, so the shipped operating point is deliberately
 * precision-lean and 0.785 restores it:
 *
 *   shipped wave3   @0.70    prec 0.7466  rec 0.7727  f1 0.7594  AUC 0.9474
 *   candidate       @0.70    prec 0.7150  rec 0.8187  f1 0.7633  AUC 0.9537
 *   candidate       @0.785   prec 0.7507  rec 0.7894  f1 0.7695  AUC 0.9537
 *
 * Same 188 false positives as shipped, 12 fewer missed closures, +0.010 f1,
 * strictly better AUC. Re-derived independently on both torch and the exported
 * ONNX, which agree exactly at 0.785 (zero gate flips).
 *
 * The pairing is not optional: the candidate trunk driven through the SHIPPED
 * head measured precision 0.4414 with 848 false closures on 4103 rows. Trunk,
 * head and threshold are one change.
 */
export function r2bGate(logit: number, threshold = 0.785): number {
  const probability = 1 / (1 + Math.exp(-logit));
  return probability > threshold ? probability : 0;
}

export function r2bBlend(
  points: Point[],
  gate: number,
  strength = 0.9,
  destination?: Point[],
): Point[] {
  const blend = Math.min(1, Math.max(0, gate)) * strength;
  const output = destination ?? points.map(([x, y]) => [x, y] as Point);
  if (blend <= 0) {
    for (let i = 0; i < points.length; i += 1) {
      const point = output[i] ?? (output[i] = [0, 0]);
      point[0] = points[i][0];
      point[1] = points[i][1];
    }
    return output;
  }
  let centerY = 0;
  for (const point of points) centerY += point[1];
  centerY /= points.length;
  for (let index = 0; index < points.length; index += 1) {
    const point = output[index] ?? (output[index] = [0, 0]);
    point[0] = points[index][0];
    point[1] = centerY + (1 - blend) * (points[index][1] - centerY)
      + blend * R2B_TEMPLATE_YOFF[index];
  }
  return output;
}

export function centerPointsX(
  points: Point[],
  targetX: number,
  destination?: Point[],
): Point[] {
  let current = 0;
  for (const point of points) current += point[0];
  current /= points.length;
  const shift = targetX - current;
  const output = destination ?? points.map(([x, y]) => [x + shift, y] as Point);
  for (let i = 0; i < points.length; i += 1) {
    const point = output[i] ?? (output[i] = [0, 0]);
    point[0] = points[i][0] + shift;
    point[1] = points[i][1];
  }
  return output;
}
