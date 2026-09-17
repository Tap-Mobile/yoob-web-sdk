import { PLANE, RES } from "../assets/serve320-bundle";
import type { Point } from "../math/geometry";
import type { RendererSpatialContract } from "./generated/runtime-tier-contract";
import {
  FULL320_RENDERER_SPATIAL_CONTRACT,
  isFull320RendererContract,
  rendererInputLength,
  rendererOutputLength,
  resolveRendererSpatialContract,
} from "./renderer-contract";

/**
 * Reusable per-frame buffers for the Serve320 render hot path.
 * Values are overwritten each frame; nothing here is retained across
 * utterance boundaries except the fixed-capacity buffers themselves.
 */
export class FrameScratch {
  readonly hostHwc = new Uint8Array(3 * PLANE);
  readonly warpedHost = new Uint8Array(3 * PLANE);
  readonly rendererContract: RendererSpatialContract;
  readonly rendererInput: Float32Array;
  readonly rendererOutput: Float32Array;
  readonly rendererPoints: Point[] = Array.from({ length: 20 }, () => [0, 0]);
  /** Immutable held renderer result; finishFrame mutates predictionBgr. */
  readonly heldPredictionBgr = new Uint8Array(3 * PLANE);
  readonly predictionBgr = new Uint8Array(3 * PLANE);
  readonly inkShifted = new Uint8Array(3 * PLANE);
  readonly alignedReference = new Float32Array(3 * PLANE);
  readonly heatmapGaussianX: Float32Array;
  readonly heatmapGaussianY: Float32Array;
  /** State captured for the one neural frame currently dispatched ahead. */
  readonly neuralHostSnapshot = new Uint8Array(3 * PLANE);
  /** Chin-warped host used only to expand an opt-in ROI result into full320. */
  readonly neuralRendererHostSnapshot?: Uint8Array;
  readonly neuralPointSnapshot: Point[] = Array.from({ length: 20 }, () => [0, 0]);
  /** Channel plane reused while packing resized BGR output. */
  readonly resizePlane = new Uint8Array(PLANE);
  /** Scratch for bilinear float intermediate. */
  readonly resizeFloat = new Float32Array(PLANE);

  // Geometry PCA / lip landmarks (20 points × 3 generations for blend steps).
  readonly geom40 = new Float32Array(40);
  readonly geom40Ref = new Float32Array(40);
  readonly pointsA: Point[] = Array.from({ length: 20 }, () => [0, 0]);
  readonly pointsB: Point[] = Array.from({ length: 20 }, () => [0, 0]);
  readonly pointsC: Point[] = Array.from({ length: 20 }, () => [0, 0]);
  readonly refPoints: Point[] = Array.from({ length: 20 }, () => [0, 0]);

  // QA9 compositor intermediate planes (all PLANE-sized).
  readonly planeA = new Float32Array(PLANE);
  readonly planeB = new Float32Array(PLANE);
  readonly planeC = new Float32Array(PLANE);
  readonly planeD = new Float32Array(PLANE);
  readonly planeE = new Float32Array(PLANE);
  readonly planeF = new Float32Array(PLANE);
  readonly planeG = new Float32Array(PLANE);
  readonly planeH = new Float32Array(PLANE);
  readonly blurTemp = new Float32Array(PLANE);
  readonly blurKernel = new Float32Array(33);
  readonly rectTemp = new Float32Array(PLANE);
  readonly morphMid = new Float32Array(PLANE);
  readonly integral = new Int32Array((RES + 1) * (RES + 1));

  private predOut = new Uint8Array(0);
  private supportOut = new Float32Array(0);
  private supportResize = new Float32Array(0);
  private jawProtectedOut = new Uint8Array(0);
  private predOutCapacity = 0;
  private supportOutCapacity = 0;
  private rgbaOut = new Uint8ClampedArray(0);
  private rgbaCapacity = 0;

  constructor(
    contract: RendererSpatialContract = FULL320_RENDERER_SPATIAL_CONTRACT,
  ) {
    this.rendererContract = resolveRendererSpatialContract(contract);
    this.rendererInput = new Float32Array(rendererInputLength(this.rendererContract));
    this.rendererOutput = new Float32Array(rendererOutputLength(this.rendererContract));
    this.heatmapGaussianX = new Float32Array(this.rendererContract.inputWidth);
    this.heatmapGaussianY = new Float32Array(this.rendererContract.inputHeight);
    if (!isFull320RendererContract(this.rendererContract)) {
      this.neuralRendererHostSnapshot = new Uint8Array(3 * PLANE);
    }
  }

  ensureTarget(width: number, height: number): {
    predBgr: Uint8Array;
    support: Float32Array;
    supportResize: Float32Array;
    jawProtected: Uint8Array;
  } {
    const pixels = width * height;
    if (pixels > this.predOutCapacity) {
      this.predOut = new Uint8Array(pixels * 3);
      this.supportOut = new Float32Array(pixels);
      this.supportResize = new Float32Array(pixels);
      this.jawProtectedOut = new Uint8Array(pixels);
      this.predOutCapacity = pixels;
    }
    return {
      predBgr: this.predOut.subarray(0, pixels * 3),
      support: this.supportOut.subarray(0, pixels),
      supportResize: this.supportResize.subarray(0, pixels),
      jawProtected: this.jawProtectedOut.subarray(0, pixels),
    };
  }

  /**
   * Straight-alpha RGBA staging for the GPU composite path.
   *
   * `new ImageData(...)` requires an exactly-sized Uint8ClampedArray, so this
   * returns a subarray rather than the whole pooled buffer, and reallocates
   * only when the mouth box grows.
   */
  ensureRgba(width: number, height: number): Uint8ClampedArray {
    const bytes = width * height * 4;
    if (bytes > this.rgbaCapacity) {
      this.rgbaOut = new Uint8ClampedArray(bytes);
      this.rgbaCapacity = bytes;
    }
    return this.rgbaOut.subarray(0, bytes);
  }
}

export function planarBgrToHwcInto(planar: Uint8Array, output: Uint8Array): void {
  for (let pixel = 0; pixel < PLANE; pixel += 1) {
    output[pixel * 3] = planar[pixel];
    output[pixel * 3 + 1] = planar[PLANE + pixel];
    output[pixel * 3 + 2] = planar[2 * PLANE + pixel];
  }
}

export function rendererOutputToBgrInto(
  output: Float32Array,
  prediction: Uint8Array,
  width = RES,
  height = RES,
): void {
  const plane = width * height;
  if (output.length !== 3 * plane || prediction.length !== 3 * plane) {
    throw new Error(`renderer output conversion shape mismatch ${output.length}/${prediction.length}`);
  }
  for (let pixel = 0; pixel < plane; pixel += 1) {
    const red = Math.min(1, Math.max(0, (output[pixel] + 1) / 2));
    const green = Math.min(1, Math.max(0, (output[plane + pixel] + 1) / 2));
    const blue = Math.min(1, Math.max(0, (output[2 * plane + pixel] + 1) / 2));
    prediction[pixel * 3] = Math.min(255, Math.floor(blue * 255 + 0.5));
    prediction[pixel * 3 + 1] = Math.min(255, Math.floor(green * 255 + 0.5));
    prediction[pixel * 3 + 2] = Math.min(255, Math.floor(red * 255 + 0.5));
  }
}

export function restoreHeldPrediction(
  held: Uint8Array,
  working: Uint8Array,
): void {
  if (held.length !== working.length) {
    throw new Error(`held prediction mismatch ${held.length}/${working.length}`);
  }
  working.set(held);
}

export function writeHeatmapsInto(
  points: Point[],
  destination: Float32Array,
  gaussianX: Float32Array,
  gaussianY: Float32Array,
  channelOffset = 7,
  sigma = 4,
  width = RES,
  height = RES,
  pointCount = points.length,
): void {
  const plane = width * height;
  if (gaussianX.length < width || gaussianY.length < height) {
    throw new Error("heatmap Gaussian scratch is too small");
  }
  const inverse = 1 / (2 * sigma * sigma);
  if (pointCount < 0 || pointCount > points.length) {
    throw new Error(`invalid renderer point count ${pointCount}`);
  }
  for (let point = 0; point < pointCount; point += 1) {
    const [centerX, centerY] = points[point];
    for (let x = 0; x < width; x += 1) {
      const delta = x - centerX;
      gaussianX[x] = Math.exp(-delta * delta * inverse);
    }
    for (let y = 0; y < height; y += 1) {
      const delta = y - centerY;
      gaussianY[y] = Math.exp(-delta * delta * inverse);
    }
    const base = (channelOffset + point) * plane;
    for (let y = 0; y < height; y += 1) {
      const scale = gaussianY[y];
      const row = base + y * width;
      for (let x = 0; x < width; x += 1) destination[row + x] = gaussianX[x] * scale;
    }
  }
}
