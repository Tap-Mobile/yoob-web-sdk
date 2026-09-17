import { PLANE, RES } from "../assets/serve320-bundle";
import type { Point } from "./geometry";

export function writeHeatmaps(
  points: Point[],
  destination: Float32Array,
  channelOffset = 7,
  sigma = 4,
): void {
  const inverse = 1 / (2 * sigma * sigma);
  const gaussianX = new Float32Array(RES);
  const gaussianY = new Float32Array(RES);
  points.forEach(([centerX, centerY], point) => {
    for (let x = 0; x < RES; x += 1) {
      const delta = x - centerX;
      gaussianX[x] = Math.exp(-delta * delta * inverse);
    }
    for (let y = 0; y < RES; y += 1) {
      const delta = y - centerY;
      gaussianY[y] = Math.exp(-delta * delta * inverse);
    }
    const base = (channelOffset + point) * PLANE;
    for (let y = 0; y < RES; y += 1) {
      const scale = gaussianY[y];
      const row = base + y * RES;
      for (let x = 0; x < RES; x += 1) destination[row + x] = gaussianX[x] * scale;
    }
  });
}
