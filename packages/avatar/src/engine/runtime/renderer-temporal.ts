import { PLANE } from "../assets/serve320-bundle";
import type { Point } from "../math/geometry";
import { roundToEven } from "../math/rounding";
import type {
  RendererSpatialContract,
  RendererTemporalContract,
} from "./generated/runtime-tier-contract";
export type { RendererTemporalContract } from "./generated/runtime-tier-contract";
import { EXPERIMENTAL_224X160_RENDERER_SPATIAL_CONTRACT } from "./renderer-contract";

/** Frozen Eval-v2 winner. Production catalogs may select only this exact contract. */
export const EXPERIMENTAL_224X160_TEMPORAL_CONTRACT = Object.freeze({
  contractVersion: "serve224x160.web.temporal-boost.v1",
  mechanism: "motionCompensatedResidualBoostV1",
  beta: 0.10,
  deltaClipU8: 12,
  warpSupportPixels: 24,
  resetPolicy: "epoch-frame-continuity-v1",
} satisfies RendererTemporalContract);

const TEMPORAL_KEYS = new Set([
  "contractVersion", "mechanism", "beta", "deltaClipU8",
  "warpSupportPixels", "resetPolicy",
]);

function sameSpatialContract(
  left: RendererSpatialContract,
  right: RendererSpatialContract,
): boolean {
  return left.baseWidth === right.baseWidth && left.baseHeight === right.baseHeight
    && left.inputWidth === right.inputWidth && left.inputHeight === right.inputHeight
    && left.originX === right.originX && left.originY === right.originY
    && left.inputChannels === right.inputChannels
    && left.signalChannels === right.signalChannels
    && left.outputChannels === right.outputChannels
    && left.outputMode === right.outputMode
    && left.roiFeatherPixels === right.roiFeatherPixels
    && left.compositeOrder === right.compositeOrder;
}

/**
 * Only the measured 224x160/beta=.10 combination is accepted. A profile is an
 * immutable experiment contract, not a bag of tuning knobs that can silently
 * create an unevaluated runtime.
 */
export function validateRendererTemporalContract(
  value: unknown,
  spatial: RendererSpatialContract | undefined,
): asserts value is RendererTemporalContract {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("renderer temporal contract must be an object");
  }
  const source = value as Record<string, unknown>;
  for (const key of Object.keys(source)) {
    if (!TEMPORAL_KEYS.has(key)) throw new Error(`renderer temporal contract has unknown ${key}`);
  }
  const selected = EXPERIMENTAL_224X160_TEMPORAL_CONTRACT;
  for (const key of TEMPORAL_KEYS) {
    if (source[key] !== selected[key as keyof RendererTemporalContract]) {
      throw new Error(`renderer temporal contract ${key} is not the frozen Web winner`);
    }
  }
  if (!spatial || !sameSpatialContract(
    spatial, EXPERIMENTAL_224X160_RENDERER_SPATIAL_CONTRACT,
  )) {
    throw new Error("renderer temporal contract requires the exact native 224x160 spatial contract");
  }
}

export function isRendererTemporalContract(
  value: unknown,
  spatial: RendererSpatialContract | undefined,
): value is RendererTemporalContract {
  try {
    validateRendererTemporalContract(value, spatial);
    return true;
  } catch {
    return false;
  }
}

export function resolveRendererTemporalContract(
  value: RendererTemporalContract | undefined,
  spatial: RendererSpatialContract,
): RendererTemporalContract | undefined {
  if (value === undefined) return undefined;
  validateRendererTemporalContract(value, spatial);
  return value;
}

export interface RendererTemporalApplication {
  handled: boolean;
  effectApplied: boolean;
}

/** Stateful raw-residual history. Boosted pixels are never fed back. */
export class RendererTemporalState {
  previousResidual: Float32Array;
  currentResidual: Float32Array;
  readonly previousPoints = new Float32Array(40);
  hasHistory = false;
  lastEpoch = -1;
  lastFrame = -1;

  constructor(readonly width: number, readonly height: number) {
    this.previousResidual = new Float32Array(width * height * 3);
    this.currentResidual = new Float32Array(width * height * 3);
  }

  reset(): void {
    this.hasHistory = false;
    this.lastEpoch = -1;
    this.lastFrame = -1;
  }
}

interface TemporalOptions {
  beta: number;
  deltaClipU8: number;
  warpSupportPixels: number;
}

function fadd(left: number, right: number): number {
  return Math.fround(Math.fround(left) + Math.fround(right));
}

function fsub(left: number, right: number): number {
  return Math.fround(Math.fround(left) - Math.fround(right));
}

function fmul(left: number, right: number): number {
  return Math.fround(Math.fround(left) * Math.fround(right));
}

function fdiv(left: number, right: number): number {
  return Math.fround(Math.fround(left) / Math.fround(right));
}

function bilinearZero(
  residual: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  channel: number,
): number {
  // OpenCV INTER_LINEAR uses a 32-entry interpolation table. Quantizing here
  // closes the evaluator/Web interpolation contract, including zero borders.
  const qx = Math.fround(roundToEven(fmul(x, 32)) / 32);
  const qy = Math.fround(roundToEven(fmul(y, 32)) / 32);
  const x0 = Math.floor(qx), y0 = Math.floor(qy);
  const x1 = x0 + 1, y1 = y0 + 1;
  const wx = fsub(qx, x0), wy = fsub(qy, y0);
  const sample = (sx: number, sy: number): number => {
    if (sx < 0 || sy < 0 || sx >= width || sy >= height) return 0;
    return residual[(sy * width + sx) * 3 + channel];
  };
  const top = fadd(fmul(sample(x0, y0), fsub(1, wx)), fmul(sample(x1, y0), wx));
  const bottom = fadd(fmul(sample(x0, y1), fsub(1, wx)), fmul(sample(x1, y1), wx));
  return fadd(fmul(top, fsub(1, wy)), fmul(bottom, wy));
}

/**
 * Deterministic CPU mirror of the frozen evaluator/iOS Metal mechanism. Inputs
 * are packed ROI BGR. The first frame, new epoch, discontinuity, and beta zero
 * are byte-identical while replacing history with the current raw residual.
 */
export function applyMotionCompensatedResidualBoost(
  candidate: Uint8Array,
  host: Uint8Array,
  activeSupport: Float32Array,
  landmarks: readonly Point[],
  epoch: number,
  frame: number,
  state: RendererTemporalState,
  options: TemporalOptions,
): RendererTemporalApplication {
  const { width, height } = state;
  const pixels = width * height;
  if (candidate.length !== pixels * 3 || host.length !== pixels * 3
      || activeSupport.length !== pixels || landmarks.length !== 20) {
    throw new Error("renderer temporal input shape mismatch");
  }
  if (!Number.isSafeInteger(epoch) || !Number.isSafeInteger(frame)) {
    throw new Error("renderer temporal epoch/frame must be safe integers");
  }
  if (!Number.isFinite(options.beta) || options.beta < 0 || options.beta > 1
      || !Number.isFinite(options.deltaClipU8) || options.deltaClipU8 <= 0
      || !Number.isFinite(options.warpSupportPixels) || options.warpSupportPixels <= 0) {
    throw new Error("renderer temporal options are invalid");
  }
  for (const point of landmarks) {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) {
      throw new Error("renderer temporal landmarks must be finite");
    }
  }

  const consecutive = state.hasHistory && state.lastEpoch === epoch
    && state.lastFrame < Number.MAX_SAFE_INTEGER && frame === state.lastFrame + 1;
  const useHistory = consecutive;
  let hasMotion = false;
  let minX = Math.fround(landmarks[0][0]);
  let minY = Math.fround(landmarks[0][1]);
  let maxX = minX, maxY = minY;
  for (let point = 0; point < 20; point += 1) {
    const destinationX = Math.fround(landmarks[point][0]);
    const destinationY = Math.fround(landmarks[point][1]);
    const sourceX = useHistory ? state.previousPoints[point * 2] : destinationX;
    const sourceY = useHistory ? state.previousPoints[point * 2 + 1] : destinationY;
    minX = Math.min(minX, sourceX, destinationX);
    minY = Math.min(minY, sourceY, destinationY);
    maxX = Math.max(maxX, sourceX, destinationX);
    maxY = Math.max(maxY, sourceY, destinationY);
    if (useHistory) {
      const dx = fsub(destinationX, sourceX);
      const dy = fsub(destinationY, sourceY);
      if (fadd(fmul(dx, dx), fmul(dy, dy)) > 1e-12) hasMotion = true;
    }
  }
  minX = Math.fround(minX); minY = Math.fround(minY);
  maxX = Math.fround(maxX); maxY = Math.fround(maxY);
  const warp = Math.fround(options.warpSupportPixels);
  const originX = EXPERIMENTAL_224X160_RENDERER_SPATIAL_CONTRACT.originX;
  const originY = EXPERIMENTAL_224X160_RENDERER_SPATIAL_CONTRACT.originY;
  const x0 = Math.max(0, Math.floor(fsub(minX, warp)));
  const y0 = Math.max(0, Math.floor(fsub(minY, warp)));
  const x1 = Math.min(320, Math.ceil(fadd(fadd(maxX, warp), 1)));
  const y1 = Math.min(320, Math.ceil(fadd(fadd(maxY, warp), 1)));

  const raw = state.currentResidual;
  for (let index = 0; index < raw.length; index += 1) {
    raw[index] = candidate[index] - host[index];
  }

  let changed = false;
  if (useHistory && options.beta > 0) {
    const beta = Math.fround(options.beta);
    const clip = Math.fround(options.deltaClipU8);
    for (let y = 0; y < height; y += 1) {
      const globalY = originY + y;
      for (let x = 0; x < width; x += 1) {
        const pixel = y * width + x;
        const base = pixel * 3;
        if (activeSupport[pixel] <= 1e-4) {
          candidate[base] = host[base];
          candidate[base + 1] = host[base + 1];
          candidate[base + 2] = host[base + 2];
          continue;
        }
        const globalX = originX + x;
        let displaceX = 0, displaceY = 0;
        let sumWeight = 0;
        let envelope = 1;
        const warpPixel = hasMotion && globalX >= x0 && globalY >= y0
          && globalX < x1 && globalY < y1;
        if (warpPixel) {
          const px = Math.fround(globalX), py = Math.fround(globalY);
          for (let point = 0; point < 20; point += 1) {
            const destinationX = Math.fround(landmarks[point][0]);
            const destinationY = Math.fround(landmarks[point][1]);
            const sourceX = state.previousPoints[point * 2];
            const sourceY = state.previousPoints[point * 2 + 1];
            const dx = fsub(px, destinationX), dy = fsub(py, destinationY);
            const denominator = fadd(fadd(fmul(dx, dx), fmul(dy, dy)), 0.25);
            const weight = fdiv(1, denominator);
            sumWeight = fadd(sumWeight, weight);
            displaceX = fadd(displaceX, fmul(weight, fsub(destinationX, sourceX)));
            displaceY = fadd(displaceY, fmul(weight, fsub(destinationY, sourceY)));
          }
          displaceX = fdiv(displaceX, sumWeight);
          displaceY = fdiv(displaceY, sumWeight);
          const outsideX = Math.fround(Math.max(Math.max(fsub(minX, px), 0), fsub(px, maxX)));
          const outsideY = Math.fround(Math.max(Math.max(fsub(minY, py), 0), fsub(py, maxY)));
          const distance = Math.fround(Math.sqrt(fadd(fmul(outsideX, outsideX), fmul(outsideY, outsideY))));
          const position = Math.fround(Math.min(Math.max(fsub(1, fdiv(distance, warp)), 0), 1));
          const polynomial = fadd(fmul(position, fadd(fmul(position, 6), -15)), 10);
          envelope = fmul(fmul(fmul(position, position), position), polynomial);
        }
        for (let channel = 0; channel < 3; channel += 1) {
          const prior = warpPixel
            ? bilinearZero(
              state.previousResidual, width, height,
              fsub(x, fmul(displaceX, envelope)),
              fsub(y, fmul(displaceY, envelope)), channel,
            )
            : state.previousResidual[base + channel];
          const delta = Math.min(clip, Math.max(-clip, fsub(raw[base + channel], prior)));
          const value = fadd(fadd(host[base + channel], raw[base + channel]), fmul(beta, delta));
          const output = Math.min(255, Math.max(0, roundToEven(value)));
          if (output !== candidate[base + channel]) changed = true;
          candidate[base + channel] = output;
        }
      }
    }
  }

  state.currentResidual = state.previousResidual;
  state.previousResidual = raw;
  for (let point = 0; point < 20; point += 1) {
    state.previousPoints[point * 2] = landmarks[point][0];
    state.previousPoints[point * 2 + 1] = landmarks[point][1];
  }
  state.hasHistory = true;
  state.lastEpoch = epoch;
  state.lastFrame = frame;
  return { handled: true, effectApplied: useHistory && options.beta > 0 && changed };
}

/**
 * Native ROI compositor bridge: capture pre-feather activity, host-fill using
 * post-feather support, optionally apply the temporal winner, then expose
 * binary ownership so the final canvas blend cannot apply support twice.
 */
export class NativeRoiPostprocessor {
  private readonly activeSupport: Float32Array;
  private readonly binarySupport: Float32Array;
  private readonly rawPrediction: Uint8Array;
  private readonly candidate: Uint8Array;
  private readonly host: Uint8Array;
  private readonly roiMultiplier: Float32Array;
  private readonly temporalState?: RendererTemporalState;

  constructor(
    readonly spatial: RendererSpatialContract,
    readonly supportMultiplier: Float32Array,
    readonly temporal?: RendererTemporalContract,
  ) {
    const pixels = spatial.inputWidth * spatial.inputHeight;
    if (supportMultiplier.length !== PLANE) {
      throw new Error("native ROI support multiplier shape mismatch");
    }
    if (temporal) validateRendererTemporalContract(temporal, spatial);
    this.activeSupport = new Float32Array(pixels);
    this.binarySupport = new Float32Array(pixels);
    this.rawPrediction = new Uint8Array(pixels * 3);
    this.candidate = new Uint8Array(pixels * 3);
    this.host = new Uint8Array(pixels * 3);
    this.roiMultiplier = new Float32Array(pixels);
    for (let y = 0; y < spatial.inputHeight; y += 1) {
      const globalRow = (spatial.originY + y) * spatial.baseWidth + spatial.originX;
      const localRow = y * spatial.inputWidth;
      for (let x = 0; x < spatial.inputWidth; x += 1) {
        this.roiMultiplier[localRow + x] = supportMultiplier[globalRow + x];
      }
    }
    this.temporalState = temporal
      ? new RendererTemporalState(spatial.inputWidth, spatial.inputHeight) : undefined;
  }

  apply(
    prediction: Uint8Array,
    rendererHost: Uint8Array,
    support: Float32Array,
    landmarks: readonly Point[],
    epoch: number,
    frame: number,
  ): RendererTemporalApplication {
    if (prediction.length !== 3 * PLANE || rendererHost.length !== 3 * PLANE
        || support.length !== PLANE) {
      throw new Error("native ROI full-canvas shape mismatch");
    }
    const { inputWidth: width, inputHeight: height, originX, originY } = this.spatial;
    for (let y = 0; y < height; y += 1) {
      const globalRow = (originY + y) * this.spatial.baseWidth + originX;
      const localRow = y * width;
      for (let x = 0; x < width; x += 1) {
        const localPixel = localRow + x;
        const globalPixel = globalRow + x;
        const localBase = localPixel * 3;
        const globalBase = globalPixel * 3;
        this.activeSupport[localPixel] = support[globalPixel];
        for (let channel = 0; channel < 3; channel += 1) {
          this.rawPrediction[localBase + channel] = prediction[globalBase + channel];
          this.host[localBase + channel] = rendererHost[globalBase + channel];
        }
      }
    }
    const packed = this.applyPacked(
      this.rawPrediction, this.host, this.activeSupport, landmarks, epoch, frame,
    );

    support.fill(0);
    for (let y = 0; y < height; y += 1) {
      const globalRow = (originY + y) * this.spatial.baseWidth + originX;
      const localRow = y * width;
      for (let x = 0; x < width; x += 1) {
        const localPixel = localRow + x;
        const globalPixel = globalRow + x;
        const localBase = localPixel * 3;
        const globalBase = globalPixel * 3;
        prediction[globalBase] = packed.prediction[localBase];
        prediction[globalBase + 1] = packed.prediction[localBase + 1];
        prediction[globalBase + 2] = packed.prediction[localBase + 2];
        support[globalPixel] = packed.support[localPixel];
      }
    }
    return packed.application;
  }

  /** Ordered pipeline entry point for output prepared by parallel QA9 workers. */
  applyPacked(
    rawPrediction: Uint8Array,
    rendererHost: Uint8Array,
    activeSupport: Float32Array,
    landmarks: readonly Point[],
    epoch: number,
    frame: number,
  ): {
    prediction: Uint8Array;
    support: Float32Array;
    application: RendererTemporalApplication;
  } {
    const pixels = this.spatial.inputWidth * this.spatial.inputHeight;
    if (rawPrediction.length !== pixels * 3 || rendererHost.length !== pixels * 3
        || activeSupport.length !== pixels) {
      throw new Error("native ROI packed input shape mismatch");
    }
    this.activeSupport.set(activeSupport);
    this.host.set(rendererHost);
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      const active = this.activeSupport[pixel];
      const weight = Math.min(1, Math.max(0, active * this.roiMultiplier[pixel]));
      const base = pixel * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        const host = this.host[base + channel];
        const value = host * (1 - weight) + rawPrediction[base + channel] * weight;
        this.candidate[base + channel] = Math.min(255, Math.max(0, Math.floor(value + 0.5)));
      }
      this.binarySupport[pixel] = active * this.roiMultiplier[pixel] > 1e-4 ? 1 : 0;
    }

    let application: RendererTemporalApplication = { handled: false, effectApplied: false };
    if (this.temporal && this.temporalState) {
      application = applyMotionCompensatedResidualBoost(
        this.candidate,
        this.host,
        this.activeSupport,
        landmarks,
        epoch,
        frame,
        this.temporalState,
        this.temporal,
      );
    }
    return {
      prediction: this.candidate,
      support: this.binarySupport,
      application,
    };
  }
}
