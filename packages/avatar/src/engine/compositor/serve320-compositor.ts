import { PLANE, RES, type Serve320Bundle } from "../assets/serve320-bundle";
import type { Point } from "../math/geometry";
import { roundToEven } from "../math/rounding";
import {
  convexHull, dilateDownBinary, dilateRect, erodeEllipse5x5, erodeRectBinary,
  fillConvexPoly, gaussianBlur, median, morphCloseRect,
  resizeBilinearFloat,
} from "./image-ops";

export interface MouthInkMetrics {
  boundsCenterX: number;
  centroidX: number;
  minX: number; maxX: number; minY: number; maxY: number;
  pixelCount: number; componentCount: number;
}

/** Optional preallocated planes for the QA9 CPU compositor hot path. */
export interface CompositorPool {
  planeA: Float32Array;
  planeB: Float32Array;
  planeC: Float32Array;
  planeD: Float32Array;
  planeE: Float32Array;
  planeF: Float32Array;
  planeG: Float32Array;
  planeH: Float32Array;
  blurTemp: Float32Array;
  blurKernel: Float32Array;
  rectTemp: Float32Array;
  morphMid: Float32Array;
  integral: Int32Array;
}

export function chinWarp(
  source: Uint8Array,
  contour: Uint8Array,
  aperture: number,
  mouthFloor = 198,
  /** Optional preallocated HWC buffer used only when a warp is applied. */
  destination?: Uint8Array,
): Uint8Array {
  const fraction = Math.min(1, Math.max(0, (aperture - 0.20) / 0.60));
  const distance = fraction * 16;
  // Identity: return the source view (read-only use sites). Avoids a 300 KB copy.
  if (distance < 0.5) return source;
  let chin = RES - 1;
  outer: for (let y = RES - 1; y >= 0; y -= 1) {
    for (let x = 0; x < RES; x += 1) if (contour[y * RES + x] >= 128) { chin = y; break outer; }
  }
  const start = Math.max(mouthFloor, chin - 28);
  if (start >= chin) return source;
  const output = destination ?? source.slice();
  if (destination) destination.set(source);
  for (let y = start; y < RES; y += 1) {
    const dy = distance * Math.min(1, Math.max(0, (y - start) / Math.max(chin - start, 1)));
    const position = Math.min(RES - 1, Math.max(0, y - dy));
    const y0 = Math.floor(position), y1 = Math.min(y0 + 1, RES - 1), weight = position - y0;
    for (let x = 0; x < RES; x += 1) for (let channel = 0; channel < 3; channel += 1) {
      const value = source[(y0 * RES + x) * 3 + channel] * (1 - weight)
        + source[(y1 * RES + x) * 3 + channel] * weight;
      output[(y * RES + x) * 3 + channel] = Math.min(255, Math.floor(value + 0.5));
    }
  }
  return output;
}

export function apertureGatedSupport(
  source: Float32Array,
  contour: Uint8Array,
  aperture: number,
  hole: Float32Array,
  pool?: CompositorPool,
) {
  const depth = roundToEven(Math.min(1, Math.max(0, (aperture - 0.20) / 0.60)) * 70);
  const inside = pool?.planeA ?? new Float32Array(PLANE);
  // Track the contour support bbox while binarising. Every downstream stage
  // (down-dilate, erode, blur, ramp) is consumed only on or near that support,
  // so each is computed inside a window derived from it — with the plane
  // zeroed outside, which is exactly what the full-plane passes produce.
  let bx0 = RES, bx1 = -1, by0 = RES, by1 = -1;
  for (let y = 0; y < RES; y += 1) {
    const row = y * RES;
    for (let x = 0; x < RES; x += 1) {
      const value = contour[row + x] >= 128 ? 1 : 0;
      inside[row + x] = value;
      if (value) {
        if (x < bx0) bx0 = x;
        if (x > bx1) bx1 = x;
        if (y < by0) by0 = y;
        if (y > by1) by1 = y;
      }
    }
  }
  const output = pool?.planeF ?? new Float32Array(PLANE);
  output.fill(0);
  if (bx1 < 0) return output;   // empty contour: the full-plane pass yields zero too
  const extended = dilateDownBinary(inside, RES, RES, depth, pool?.planeB, {
    x0: bx0, x1: bx1, y0: by0, y1: Math.min(RES - 1, by1 + depth),
  });
  const binary = pool?.planeC ?? new Float32Array(PLANE);
  binary.fill(0);
  // The dilate's support, one stage narrower than its window.
  let sx0 = RES, sx1 = -1, sy0 = RES, sy1 = -1;
  for (let y = by0; y <= Math.min(RES - 1, by1 + depth); y += 1) {
    const row = y * RES;
    for (let x = bx0; x <= bx1; x += 1) {
      const value = extended[row + x] >= 0.5 ? 1 : 0;
      binary[row + x] = value;
      if (value) {
        if (x < sx0) sx0 = x;
        if (x > sx1) sx1 = x;
        if (y < sy0) sy0 = y;
        if (y > sy1) sy1 = y;
      }
    }
  }
  if (sx1 < 0) return output;   // no support: all zero, as the full pass yields
  // The blur's horizontal pass reads the erosion through reflect101, which at
  // the plane borders reaches beyond the naive [support ± radius] box. The
  // erosion window covers the exact reflected column/row range instead.
  const blurRadius = Math.floor((roundToEven(1.2 * 8 + 1) | 1) / 2);
  const ex0 = Math.max(0, Math.min(sx0 - blurRadius, 2 * (RES - 1) - (sx1 + blurRadius)));
  const ex1 = Math.min(RES - 1, Math.max(sx1 + blurRadius, blurRadius - sx0));
  const ey0 = Math.max(0, Math.min(sy0 - blurRadius, 2 * (RES - 1) - (sy1 + blurRadius)));
  const ey1 = Math.min(RES - 1, Math.max(sy1 + blurRadius, blurRadius - sy0));
  const eroded = erodeEllipse5x5(
    binary, RES, RES, pool?.planeD, pool?.blurTemp, pool?.morphMid, {
      x0: ex0, y0: ey0, x1: ex1, y1: ey1,
    },
  );
  const blurred = gaussianBlur(
    eroded, RES, RES, 1.2, pool?.planeE, pool?.blurTemp, pool?.blurKernel,
    { x0: sx0, y0: sy0, x1: sx1, y1: sy1 },
  );
  for (let y = sy0; y <= sy1; y += 1) {
    const row = y * RES;
    for (let x = sx0; x <= sx1; x += 1) {
      const i = row + x;
      let ramp = Math.min(blurred[i], binary[i]);
      if (hole[i] >= 0.5) ramp = binary[i];
      output[i] = binary[i] < 0.5 ? 0 : Math.min(source[i], ramp);
    }
  }
  return output;
}

function wholeCropChannel(host: Uint8Array, channel: number): number[] {
  const values = new Array<number>(PLANE);
  for (let i = 0; i < PLANE; i += 1) values[i] = host[i * 3 + channel];
  return values;
}

export function capDcCorrect(
  prediction: Uint8Array, host: Uint8Array, lipY: number, capY: number,
  mask: Float32Array, centerX: number,
): void {
  const cap = roundToEven(capY), lip = roundToEven(lipY), half = Math.trunc(0.22 * RES);
  const y0 = cap, y1 = Math.min(RES, cap + 14);
  const x0 = Math.max(0, Math.trunc(centerX) - half), x1 = Math.min(RES, Math.trunc(centerX) + half);
  if (y1 <= y0 || x1 <= x0) return;
  const hostValues = [[], [], []] as number[][];
  const predValues = [[], [], []] as number[][];
  for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) {
    const pixel = y * RES + x, base = pixel * 3;
    if (mask[pixel] <= 0.1 || Math.max(host[base], host[base + 1], host[base + 2]) <= 120) continue;
    for (let channel = 0; channel < 3; channel += 1) {
      hostValues[channel].push(host[base + channel]); predValues[channel].push(prediction[base + channel]);
    }
  }
  if (hostValues[0].length < 30) return;
  const delta = [0, 1, 2].map((channel) => Math.min(35, Math.max(-35,
    median(hostValues[channel]) - median(predValues[channel]))));
  if (delta[0] === 0 && delta[1] === 0 && delta[2] === 0) return;
  const denominator = Math.max(cap - lip, 1);
  // taper <= 0 for y <= lip, so those rows are skipped by the original loop —
  // starting at lip + 1 is the same iteration set.
  for (let y = lip + 1; y < RES; y += 1) {
    const taper = Math.min(1, Math.max(0, (y - lip) / denominator));
    for (let x = 0; x < RES; x += 1) for (let channel = 0; channel < 3; channel += 1) {
      const index = (y * RES + x) * 3 + channel;
      prediction[index] = Math.trunc(Math.min(255, Math.max(0, prediction[index] + delta[channel] * taper)));
    }
  }
}

export function hostSkinMask(
  host: Uint8Array,
  centerX: number,
  capY: number,
  pool?: CompositorPool,
) {
  const cap = roundToEven(capY), half = Math.trunc(0.22 * RES);
  const values = [[], [], []] as number[][];
  for (let y = cap; y < Math.min(RES, cap + 14); y += 1) {
    for (let x = Math.max(0, Math.trunc(centerX) - half);
      x < Math.min(RES, Math.trunc(centerX) + half); x += 1) {
      const base = (y * RES + x) * 3;
      if (Math.max(host[base], host[base + 1], host[base + 2]) <= 120) continue;
      for (let channel = 0; channel < 3; channel += 1) values[channel].push(host[base + channel]);
    }
  }
  const reference = values[0].length > 30
    ? values.map((channel) => median(channel))
    : [0, 1, 2].map((channel) => median(wholeCropChannel(host, channel)));
  const mask = pool?.planeG ?? new Float32Array(PLANE);
  if (pool) mask.fill(0);
  // The reference is a median of u8 samples, so 2*reference is always an
  // integer and the squared distance S = (2db)^2+(2dg)^2+(2dr)^2 is an exact
  // integer in f64. S != 14400 decides the Math.hypot(db,dg,dr) < 60 test
  // outright (the true sqrt is then at least 1/4 away from 60 in squared
  // terms, far beyond hypot's ~1-ulp error). S == 14400 must reproduce
  // hypot's exact rounding — it can land one ulp under 60 — so that rare
  // boundary pixel goes through the original f64 call.
  const ref2 = [2 * reference[0], 2 * reference[1], 2 * reference[2]];
  let sx0 = RES, sx1 = -1, sy0 = RES, sy1 = -1;
  for (let y = 0; y < RES; y += 1) {
    const row = y * RES;
    for (let x = 0; x < RES; x += 1) {
      const base = (row + x) * 3;
      const db = 2 * host[base] - ref2[0];
      const dg = 2 * host[base + 1] - ref2[1];
      const dr = 2 * host[base + 2] - ref2[2];
      const squared = db * db + dg * dg + dr * dr;
      const inside = squared < 14400
        || (squared === 14400
          && Math.hypot(
            host[base] - reference[0],
            host[base + 1] - reference[1],
            host[base + 2] - reference[2],
          ) < 60);
      if (inside) {
        mask[row + x] = 1;
        if (x < sx0) sx0 = x;
        if (x > sx1) sx1 = x;
        if (y < sy0) sy0 = y;
        if (y > sy1) sy1 = y;
      }
    }
  }
  if (sx1 < 0) {
    const empty = pool?.planeA ?? new Float32Array(PLANE);
    empty.fill(0);
    return empty;   // empty close stays empty
  }
  const closeRadius = Math.floor(5 / 2);
  return morphCloseRect(
    mask, RES, RES, 5, pool?.planeA, pool?.rectTemp, pool?.morphMid, {
      x0: Math.max(0, sx0 - closeRadius),
      y0: Math.max(0, sy0 - closeRadius),
      x1: Math.min(RES - 1, sx1 + closeRadius),
      y1: Math.min(RES - 1, sy1 + closeRadius),
    },
  );
}

export function hostMouthResidualMask(
  skin: Float32Array, hole: Float32Array, contour: Uint8Array,
  center: Point, lowerLipY: number, anchorWidth: number,
  destination?: Float32Array,
) {
  const radiusX = 0.56 * anchorWidth;
  const top = center[1] - 0.16 * anchorWidth;
  const bottom = lowerLipY + 0.035 * anchorWidth;
  const ellipseY = (top + bottom) * 0.5, radiusY = Math.max((bottom - top) * 0.5, 1);
  const output = destination ?? new Float32Array(PLANE);
  if (destination) output.fill(0);
  for (let y = Math.max(0, Math.floor(top)); y < Math.min(RES, Math.ceil(bottom) + 1); y += 1) {
    for (let x = Math.max(0, Math.floor(center[0] - radiusX));
      x < Math.min(RES, Math.ceil(center[0] + radiusX) + 1); x += 1) {
      const pixel = y * RES + x;
      const nx = (x - center[0]) / radiusX, ny = (y - ellipseY) / radiusY;
      if (nx * nx + ny * ny > 1 || hole[pixel] < 0.5 || skin[pixel] >= 0.5) continue;
      let safe = true;
      for (const dy of [-8, 0, 8]) for (const dx of [-8, 0, 8]) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= RES || yy < 0 || yy >= RES || contour[yy * RES + xx] < 128) safe = false;
      }
      if (safe) output[pixel] = 1;
    }
  }
  return output;
}

function percentile(histogram: Int32Array, count: number, q: number) {
  const position = q * (count - 1), low = Math.floor(position), high = Math.ceil(position);
  const value = (rank: number) => {
    let seen = 0;
    for (let x = 0; x < histogram.length; x += 1) {
      if (rank < seen + histogram[x]) return x;
      seen += histogram[x];
    }
    return 0;
  };
  return value(low) + (value(high) - value(low)) * (position - low);
}

export function mouthInkMetrics(
  bytes: Uint8Array, width: number, height: number, expectedX: number, expectedY: number,
): MouthInkMetrics | undefined {
  const x0 = Math.max(0, Math.ceil(expectedX - 70)), x1 = Math.min(width - 1, Math.floor(expectedX + 70));
  const y0 = Math.max(0, Math.ceil(expectedY - 30)), y1 = Math.min(height - 1, Math.floor(expectedY + 48));
  const localWidth = x1 - x0 + 1, localHeight = y1 - y0 + 1;
  const ink = new Uint8Array(localWidth * localHeight);
  for (let ly = 0; ly < localHeight; ly += 1) for (let lx = 0; lx < localWidth; lx += 1) {
    const base = ((y0 + ly) * width + x0 + lx) * 3;
    const b = bytes[base], g = bytes[base + 1], r = bytes[base + 2];
    if ((r - g > 14 && r - b > 8 && g < 215) || (r < 145 && g < 105 && b < 120)) {
      ink[ly * localWidth + lx] = 1;
    }
  }
  const seen = new Uint8Array(ink.length), histogram = new Int32Array(width);
  let count = 0, components = 0, minX = width, maxX = 0, minY = height, maxY = 0, xSum = 0;
  for (let seed = 0; seed < ink.length; seed += 1) {
    if (!ink[seed] || seen[seed]) continue;
    const queue = [seed]; seen[seed] = 1;
    let head = 0, cMinX = width, cMaxX = 0, cMinY = height, cMaxY = 0;
    while (head < queue.length) {
      const index = queue[head++], ly = Math.floor(index / localWidth), lx = index % localWidth;
      const x = x0 + lx, y = y0 + ly;
      cMinX = Math.min(cMinX, x); cMaxX = Math.max(cMaxX, x);
      cMinY = Math.min(cMinY, y); cMaxY = Math.max(cMaxY, y);
      for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        if (!dx && !dy) continue;
        const nx = lx + dx, ny = ly + dy;
        if (nx < 0 || nx >= localWidth || ny < 0 || ny >= localHeight) continue;
        const next = ny * localWidth + nx;
        if (ink[next] && !seen[next]) { seen[next] = 1; queue.push(next); }
      }
    }
    if (queue.length < 8 || cMaxX - cMinX < 5 ||
        Math.abs((cMinX + cMaxX) * 0.5 - expectedX) > 45 ||
        Math.abs((cMinY + cMaxY) * 0.5 - expectedY) > 30) continue;
    components += 1; minX = Math.min(minX, cMinX); maxX = Math.max(maxX, cMaxX);
    minY = Math.min(minY, cMinY); maxY = Math.max(maxY, cMaxY);
    for (const index of queue) { const x = x0 + index % localWidth; histogram[x] += 1; xSum += x; count += 1; }
  }
  if (!count) return undefined;
  return {
    boundsCenterX: (percentile(histogram, count, 0.01) + percentile(histogram, count, 0.99)) * 0.5,
    centroidX: xSum / count, minX, maxX, minY, maxY, pixelCount: count, componentCount: components,
  };
}

export function correctMouthInk(
  prediction: Uint8Array,
  host: Uint8Array,
  center: Point,
  limit = 4,
  /** Optional preallocated buffer used only when a non-zero shift is applied. */
  shiftedDestination?: Uint8Array,
) {
  const hostMetrics = mouthInkMetrics(host, RES, RES, center[0], center[1]);
  const predMetrics = mouthInkMetrics(prediction, RES, RES, center[0], center[1]);
  if (!hostMetrics || !predMetrics) return prediction;
  // Current product default (Serve320Pipeline.swift) is centroid correction.
  // The older guarded bounds-and-centroid rule skipped asymmetric open mouths.
  const request = hostMetrics.centroidX - predMetrics.centroidX;
  const shift = Math.min(limit, Math.max(-limit, roundToEven(request)));
  if (!shift) return prediction;
  const output = shiftedDestination ?? new Uint8Array(prediction.length);
  for (let y = 0; y < RES; y += 1) for (let x = 0; x < RES; x += 1) {
    const sourceX = Math.min(RES - 1, Math.max(0, x - shift));
    for (let channel = 0; channel < 3; channel += 1) {
      output[(y * RES + x) * 3 + channel] = prediction[(y * RES + sourceX) * 3 + channel];
    }
  }
  return output;
}

export interface FinishFrameBuffers {
  predBgr: Uint8Array;
  support: Float32Array;
  /** Optional full-320 ROI edge mask, multiplied after every QA9 support gate. */
  supportMultiplier?: Float32Array;
  /**
   * Optional native-ROI bridge. It runs after every QA9 support gate and before
   * resize, with support still unfeathered. The bridge owns ROI feathering,
   * host-fill, temporal correction, and conversion to binary ownership.
   */
  nativeRoiPostprocess?: (prediction: Uint8Array, support: Float32Array) => void;
  /** Capture-only phase for ordered temporal finalization; skips two resizes. */
  deferOutputResize?: boolean;
  /** Target-space mask whose pixels must remain byte-identical to the host. */
  jawProtected?: Uint8Array;
  /** Optional destination for support bilinear resize (target pixels). */
  supportResize?: Float32Array;
  /** Optional plane scratch for channel extract (PLANE bytes). */
  plane?: Uint8Array;
  /** Optional QA9 intermediate pool. */
  pool?: CompositorPool;
}

export function buildTargetJawProtectedMask(
  hostFace: Float32Array,
  jawInterior: Float32Array,
  hostCenterY: number,
  targetWidth: number,
  targetHeight: number,
  destination?: Uint8Array,
): Uint8Array {
  const targetPixels = targetWidth * targetHeight;
  const output = destination ?? new Uint8Array(targetPixels);
  if (output.length < targetPixels) {
    throw new Error(`jaw protected buffer too small (${output.length})`);
  }
  output.fill(0, 0, targetPixels);
  for (let y = 0; y < targetHeight; y += 1) {
    // This is the exact nearest target->crop mapping used by the Swift
    // jawProtectedSnapshot/restore pair.
    const cropY = Math.min(
      RES - 1,
      Math.max(0, Math.trunc((y + 0.5) * RES / targetHeight)),
    );
    if (cropY < hostCenterY) continue;
    for (let x = 0; x < targetWidth; x += 1) {
      const cropX = Math.min(
        RES - 1,
        Math.max(0, Math.trunc((x + 0.5) * RES / targetWidth)),
      );
      const cropPixel = cropY * RES + cropX;
      if (hostFace[cropPixel] >= 0.5 && jawInterior[cropPixel] < 0.5) {
        output[y * targetWidth + x] = 1;
      }
    }
  }
  return output.subarray(0, targetPixels);
}

/**
 * Resize one HWC channel with the same half-pixel bilinear contract as
 * `resizeBilinearU8`, writing straight into an HWC destination. Avoids the
 * planar extract → float convert → pack round-trip.
 */
export function resizeBilinearU8HwcChannel(
  sourceHwc: Uint8Array,
  channel: number,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  destinationHwc: Uint8Array,
): void {
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = (y + 0.5) * scaleY - 0.5;
    let y0 = Math.floor(sourceY);
    let weightY = sourceY - y0;
    if (y0 < 0) { y0 = 0; weightY = 0; }
    if (y0 >= sourceHeight - 1) { y0 = sourceHeight - 1; weightY = 0; }
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = (x + 0.5) * scaleX - 0.5;
      let x0 = Math.floor(sourceX);
      let weightX = sourceX - x0;
      if (x0 < 0) { x0 = 0; weightX = 0; }
      if (x0 >= sourceWidth - 1) { x0 = sourceWidth - 1; weightX = 0; }
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const sample = (yy: number, xx: number) => sourceHwc[(yy * sourceWidth + xx) * 3 + channel];
      const top = sample(y0, x0) * (1 - weightX) + sample(y0, x1) * weightX;
      const bottom = sample(y1, x0) * (1 - weightX) + sample(y1, x1) * weightX;
      // Match resizeBilinearFloat's Float32Array store before u8 quantize.
      const value = Math.fround(top * (1 - weightY) + bottom * weightY);
      destinationHwc[(y * targetWidth + x) * 3 + channel] =
        Math.min(255, Math.max(0, Math.floor(value + 0.5)));
    }
  }
}

/**
 * Three-channel equivalent of resizeBilinearU8HwcChannel. Coordinates and
 * interpolation weights are shared while channel arithmetic and Float32
 * rounding remain byte-identical to the three-pass implementation.
 */
export function resizeBilinearU8Hwc3(
  sourceHwc: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  destinationHwc: Uint8Array,
): void {
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = (y + 0.5) * scaleY - 0.5;
    let y0 = Math.floor(sourceY);
    let weightY = sourceY - y0;
    if (y0 < 0) { y0 = 0; weightY = 0; }
    if (y0 >= sourceHeight - 1) { y0 = sourceHeight - 1; weightY = 0; }
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    const row0 = y0 * sourceWidth;
    const row1 = y1 * sourceWidth;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = (x + 0.5) * scaleX - 0.5;
      let x0 = Math.floor(sourceX);
      let weightX = sourceX - x0;
      if (x0 < 0) { x0 = 0; weightX = 0; }
      if (x0 >= sourceWidth - 1) { x0 = sourceWidth - 1; weightX = 0; }
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const topLeft = (row0 + x0) * 3;
      const topRight = (row0 + x1) * 3;
      const bottomLeft = (row1 + x0) * 3;
      const bottomRight = (row1 + x1) * 3;
      const inverseX = 1 - weightX;
      const destination = (y * targetWidth + x) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = sourceHwc[topLeft + channel] * inverseX
          + sourceHwc[topRight + channel] * weightX;
        const bottom = sourceHwc[bottomLeft + channel] * inverseX
          + sourceHwc[bottomRight + channel] * weightX;
        const value = Math.fround(top * (1 - weightY) + bottom * weightY);
        destinationHwc[destination + channel] =
          Math.min(255, Math.max(0, Math.floor(value + 0.5)));
      }
    }
  }
}

export function finishFrame(
  predictionInput: Uint8Array, host: Uint8Array, contour: Uint8Array,
  points: Point[], aperture: number, hostCenter: Point, anchorWidth: number,
  hostLipY: number, bundle: Serve320Bundle, targetWidth: number, targetHeight: number,
  buffers?: FinishFrameBuffers,
  shiftedInk?: Uint8Array,
) {
  const pool = buffers?.pool;
  const prediction = correctMouthInk(
    predictionInput, host, hostCenter, 4, shiftedInk,
  );
  const support = apertureGatedSupport(
    bundle.support, contour, aperture, bundle.hole, pool,
  );
  let lowerLipY = 0, centerX = 0;
  for (const point of points) { lowerLipY = Math.max(lowerLipY, point[1]); centerX += point[0]; }
  centerX /= points.length;
  const capBase = Math.max(lowerLipY, hostLipY);
  capDcCorrect(prediction, host, lowerLipY, capBase + 4, support, centerX);
  // hostSkinMask reuses planeG → planeA via morphClose; keep support in planeF.
  const skin = hostSkinMask(host, centerX, capBase, pool);
  const mouthSeed = fillConvexPoly(convexHull(points), RES, RES, pool?.planeB);
  // The dilated mouth mask lives within the hull dilated by the kernel radius;
  // everything outside is zero, so the dilate runs in that window only.
  let mx0 = RES, mx1 = -1, my0 = RES, my1 = -1;
  for (const [px, py] of points) {
    if (px < mx0) mx0 = px;
    if (px > mx1) mx1 = px;
    if (py < my0) my0 = py;
    if (py > my1) my1 = py;
  }
  const mouthRadius = Math.floor(21 / 2);
  const mouth = dilateRect(mouthSeed, RES, RES, 21, pool?.planeC, pool?.rectTemp, {
    x0: Math.max(0, Math.floor(mx0) - mouthRadius),
    y0: Math.max(0, Math.floor(my0) - mouthRadius),
    x1: Math.min(RES - 1, Math.ceil(mx1) + mouthRadius),
    y1: Math.min(RES - 1, Math.ceil(my1) + mouthRadius),
  });
  const oldMouth = hostMouthResidualMask(
    skin, bundle.hole, contour, hostCenter, hostLipY, anchorWidth, pool?.planeD,
  );
  const hostFace = pool?.planeE ?? new Float32Array(PLANE);
  // Track the face bbox for the erode: its output support is inside the face.
  let fx0 = RES, fx1 = -1, fy0 = RES, fy1 = -1;
  for (let y = 0; y < RES; y += 1) {
    const row = y * RES;
    for (let x = 0; x < RES; x += 1) {
      const value = contour[row + x] >= 128 ? 1 : 0;
      hostFace[row + x] = value;
      if (value) {
        if (x < fx0) fx0 = x;
        if (x > fx1) fx1 = x;
        if (y < fy0) fy0 = y;
        if (y > fy1) fy1 = y;
      }
    }
  }
  const jawInterior = fx1 < 0
    ? (pool?.planeH ?? new Float32Array(PLANE)).fill(0)
    : erodeRectBinary(
      hostFace, RES, RES, 11, pool?.planeH, pool?.integral,
      { x0: fx0, y0: fy0, x1: fx1, y1: fy1 },
    );
  for (let y = 0; y < RES; y += 1) {
    const capWeight = Math.min(1, Math.max(0, 1 - (y - (capBase + 4)) / 14));
    for (let x = 0; x < RES; x += 1) {
      const pixel = y * RES + x;
      support[pixel] *= capWeight * Math.max(skin[pixel], mouth[pixel], oldMouth[pixel]) * jawInterior[pixel];
    }
  }
  if (buffers?.nativeRoiPostprocess) {
    buffers.nativeRoiPostprocess(prediction, support);
  } else if (buffers?.supportMultiplier) {
    if (buffers.supportMultiplier.length !== PLANE) {
      throw new Error(`finishFrame support multiplier mismatch ${buffers.supportMultiplier.length}`);
    }
    for (let pixel = 0; pixel < PLANE; pixel += 1) {
      support[pixel] *= buffers.supportMultiplier[pixel];
    }
  }
  const jawProtected = buildTargetJawProtectedMask(
    hostFace,
    jawInterior,
    hostCenter[1],
    targetWidth,
    targetHeight,
    buffers?.jawProtected,
  );
  if (buffers?.deferOutputResize) {
    return {
      predBgr: (buffers.predBgr).subarray(0, 0),
      support: (buffers.support).subarray(0, 0),
      jawProtected,
      width: targetWidth,
      height: targetHeight,
    };
  }
  const targetPixels = targetWidth * targetHeight;
  const predBgr = buffers?.predBgr ?? new Uint8Array(targetPixels * 3);
  if (predBgr.length < targetPixels * 3) {
    throw new Error(`finishFrame predBgr buffer too small (${predBgr.length})`);
  }
  resizeBilinearU8Hwc3(
    prediction, RES, RES, targetWidth, targetHeight, predBgr,
  );
  const supportOut = buffers?.support ?? new Float32Array(targetPixels);
  if (supportOut.length < targetPixels) {
    throw new Error(`finishFrame support buffer too small (${supportOut.length})`);
  }
  const resizedSupport = resizeBilinearFloat(
    support, RES, RES, targetWidth, targetHeight,
    buffers?.supportResize ?? supportOut,
  );
  if (buffers?.supportResize) supportOut.set(resizedSupport);
  return {
    predBgr: predBgr.subarray(0, targetPixels * 3),
    support: supportOut.subarray(0, targetPixels),
    jawProtected,
    width: targetWidth,
    height: targetHeight,
  };
}

/** Exact one-pass native blend over an RGBA canvas region. Prediction is BGR. */
export function canonicalBlendRgba(
  region: Uint8ClampedArray,
  predBgr: Uint8Array,
  support: Float32Array,
  jawProtected?: Uint8Array,
): void {
  if (region.length !== support.length * 4 || predBgr.length !== support.length * 3) {
    throw new Error("canonical composite shape mismatch");
  }
  if (jawProtected && jawProtected.length !== support.length) {
    throw new Error("jaw protected mask shape mismatch");
  }
  for (let pixel = 0; pixel < support.length; pixel += 1) {
    if (jawProtected?.[pixel]) continue;
    const weight = support[pixel];
    if (weight <= 0) continue;
    const rgba = pixel * 4;
    const bgr = pixel * 3;
    const inverse = Math.fround(1 - weight);
    const red = Math.fround(
      Math.fround(region[rgba] * inverse) + Math.fround(predBgr[bgr + 2] * weight),
    );
    const green = Math.fround(
      Math.fround(region[rgba + 1] * inverse) + Math.fround(predBgr[bgr + 1] * weight),
    );
    const blue = Math.fround(
      Math.fround(region[rgba + 2] * inverse) + Math.fround(predBgr[bgr] * weight),
    );
    region[rgba] = Math.min(255, Math.max(0, roundToEven(red)));
    region[rgba + 1] = Math.min(255, Math.max(0, roundToEven(green)));
    region[rgba + 2] = Math.min(255, Math.max(0, roundToEven(blue)));
  }
}
