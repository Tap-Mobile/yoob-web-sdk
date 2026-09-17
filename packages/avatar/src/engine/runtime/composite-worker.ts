/// <reference lib="webworker" />
import type { Serve320Bundle } from "../assets/serve320-bundle";
import { finishFrame } from "../compositor/serve320-compositor";
import type { Point } from "../math/geometry";
import { FrameScratch } from "./frame-scratch";
import type { RendererSpatialContract } from "./generated/runtime-tier-contract";
import {
  isFull320RendererContract,
  resolveRendererSpatialContract,
} from "./renderer-contract";
import {
  NativeRoiPostprocessor,
  resolveRendererTemporalContract,
  type RendererTemporalContract,
} from "./renderer-temporal";

// A stateless-per-utterance QA9 compositor worker. finishFrame is byte-identical
// wherever it runs; fanning it across a worker pool overlaps the 57 ms/frame CPU
// compositor (the pipeline's dominant cost) with the serial WebGPU renderer.
// Only the constant support/hole planes are held here; every frame's host,
// contour, prediction, and geometry arrive per job.

export interface CompositeInit {
  type: "init";
  support: ArrayBuffer;
  hole: ArrayBuffer;
  supportMultiplier?: ArrayBuffer;
  rendererSpatialContract?: RendererSpatialContract;
  rendererTemporalContract?: RendererTemporalContract;
}

export interface CompositeJob {
  type: "job";
  jobId: number;
  epoch: number;
  index: number;
  predBgr: ArrayBuffer; // 3 * PLANE u8, BGR HWC (renderer output already tonemapped)
  host: ArrayBuffer; // 3 * PLANE u8, BGR HWC (unwarped idle host crop)
  rendererHost?: ArrayBuffer; // 3 * PLANE u8, BGR HWC (chin-warped ROI host)
  contour: ArrayBuffer; // PLANE u8
  points: Float64Array; // 40 = 20 * (x, y); float64 preserves exact landmark coords
  aperture: number;
  hostCenterX: number;
  hostCenterY: number;
  anchorWidth: number;
  hostLipY: number;
  box: [number, number, number, number];
  renderMs: number;
}

export interface CompositeReady { type: "ready" }
export interface CompositeResult {
  type: "result";
  jobId: number;
  index: number;
  box: [number, number, number, number];
  width: number;
  height: number;
  predBgr: ArrayBuffer;
  support: ArrayBuffer;
  jawProtected: ArrayBuffer;
  /** Corrected full canvas and final QA9 activity before ROI feather. */
  preparedPrediction?: ArrayBuffer;
  roiSupport?: ArrayBuffer;
  /**
   * Straight-alpha RGBA of the mouth region, ready for `drawImage`.
   *
   * The main thread used to do the alpha blend itself: getImageData on the
   * mouth box, canonicalBlendRgba per pixel, putImageData back. Measured at a
   * 260x340 box that is 5.31 ms/frame of main-thread work, of which the
   * per-pixel blend alone is 4.41 ms. Handing the GPU a straight-alpha bitmap
   * and letting source-over do the same arithmetic costs 0.007 ms on the main
   * thread, and the 1.0 ms pack happens here, off the critical path.
   *
   * Absent when the worker could not build one, so the caller keeps the
   * canonical CPU path as a fallback.
   */
  bitmap?: ImageBitmap;
  renderMs: number;
  compositorMs: number;
  packMs: number;
}

type Incoming = CompositeInit | CompositeJob;

const scope = self as unknown as DedicatedWorkerGlobalScope;
const scratch = new FrameScratch();
const points: Point[] = Array.from({ length: 20 }, () => [0, 0]);
let bundleShim: Serve320Bundle | undefined;
let rendererSupportMultiplier: Float32Array | undefined;
let nativeRoiPostprocessor: NativeRoiPostprocessor | undefined;
let rendererSpatialContract: RendererSpatialContract | undefined;
let rendererTemporalContract: RendererTemporalContract | undefined;

scope.onmessage = async (event: MessageEvent<Incoming>) => {
  const message = event.data;
  if (message.type === "init") {
    const support = new Float32Array(message.support);
    const hole = new Float32Array(message.hole);
    rendererSupportMultiplier = message.supportMultiplier
      ? new Float32Array(message.supportMultiplier) : undefined;
    const spatial = resolveRendererSpatialContract(message.rendererSpatialContract);
    const temporal = resolveRendererTemporalContract(
      message.rendererTemporalContract,
      spatial,
    );
    rendererSpatialContract = spatial;
    rendererTemporalContract = temporal;
    if (!isFull320RendererContract(spatial)) {
      if (!rendererSupportMultiplier) {
        throw new Error("native ROI compositor requires its support multiplier");
      }
      // Stateful temporal history belongs to the ordered pipeline worker. The
      // parallel workers only capture corrected pixels + unfeathered QA9 mask.
      nativeRoiPostprocessor = temporal ? undefined : new NativeRoiPostprocessor(
        spatial, rendererSupportMultiplier,
      );
    } else {
      nativeRoiPostprocessor = undefined;
    }
    // finishFrame only reads bundle.support and bundle.hole.
    bundleShim = { support, hole } as unknown as Serve320Bundle;
    scope.postMessage({ type: "ready" } satisfies CompositeReady);
    return;
  }
  if (!bundleShim) throw new Error("composite worker received a job before init");
  const predBgr = new Uint8Array(message.predBgr);
  const host = new Uint8Array(message.host);
  const rendererHost = message.rendererHost
    ? new Uint8Array(message.rendererHost) : undefined;
  const contour = new Uint8Array(message.contour);
  for (let i = 0; i < 20; i += 1) {
    points[i][0] = message.points[i * 2];
    points[i][1] = message.points[i * 2 + 1];
  }
  const width = message.box[2] - message.box[0];
  const height = message.box[3] - message.box[1];
  const targets = scratch.ensureTarget(width, height);
  let preparedPrediction: Uint8Array | undefined;
  let roiSupport: Float32Array | undefined;
  const compositorStarted = performance.now();
  const finished = finishFrame(
    predBgr, host, contour, points, message.aperture,
    [message.hostCenterX, message.hostCenterY] as Point,
    message.anchorWidth, message.hostLipY, bundleShim, width, height,
    {
      predBgr: targets.predBgr,
      support: targets.support,
      supportMultiplier: rendererSupportMultiplier,
      nativeRoiPostprocess: nativeRoiPostprocessor
        ? (prediction, support) => {
          if (!rendererHost) throw new Error("native ROI frame is missing its renderer host");
          nativeRoiPostprocessor!.apply(
            prediction, rendererHost, support, points, message.epoch, message.index,
          );
        }
        : rendererTemporalContract && rendererSpatialContract
          ? (prediction, support) => {
            const spatial = rendererSpatialContract!;
            const pixels = spatial.inputWidth * spatial.inputHeight;
            preparedPrediction = new Uint8Array(prediction);
            roiSupport = new Float32Array(pixels);
            for (let y = 0; y < spatial.inputHeight; y += 1) {
              const globalRow = (spatial.originY + y) * spatial.baseWidth + spatial.originX;
              const localRow = y * spatial.inputWidth;
              for (let x = 0; x < spatial.inputWidth; x += 1) {
                const localPixel = localRow + x;
                const globalPixel = globalRow + x;
                roiSupport![localPixel] = support[globalPixel];
              }
            }
          }
        : undefined,
      deferOutputResize: rendererTemporalContract !== undefined,
      supportResize: targets.supportResize,
      jawProtected: targets.jawProtected,
      plane: scratch.resizePlane,
      pool: scratch,
    },
    scratch.inkShifted,
  );
  const compositorMs = performance.now() - compositorStarted;
  // Fresh transferable copies: the pooled target buffers must not be detached.
  const predOut = new Uint8Array(finished.predBgr);
  const supportOut = new Float32Array(finished.support);
  const jawOut = new Uint8Array(finished.jawProtected);

  // Pack straight-alpha RGBA so the GPU can do the blend with source-over,
  // which computes the same dst*(1-a) + src*a the CPU path does. BGR -> RGB,
  // alpha from the support mask, jaw-protected pixels fully transparent so the
  // host shows through untouched exactly as `continue` did.
  const packStarted = performance.now();
  let bitmap: ImageBitmap | undefined;
  if (!preparedPrediction || !roiSupport) {
    try {
      const rgba = scratch.ensureRgba(width, height);
      for (let pixel = 0, o = 0, b = 0; pixel < supportOut.length; pixel += 1, o += 4, b += 3) {
        const weight = jawOut[pixel] ? 0 : supportOut[pixel];
        rgba[o] = predOut[b + 2];
        rgba[o + 1] = predOut[b + 1];
        rgba[o + 2] = predOut[b];
        rgba[o + 3] = weight <= 0 ? 0 : (weight >= 1 ? 255 : (weight * 255) | 0);
      }
      // TS models a subarray's backing store as ArrayBufferLike, which admits
      // SharedArrayBuffer; ImageData requires a plain ArrayBuffer. The pool
      // never allocates shared memory, so the narrowing is sound.
      const pixels = rgba as unknown as Uint8ClampedArray<ArrayBuffer>;
      bitmap = await createImageBitmap(new ImageData(pixels, width, height));
    } catch {
      bitmap = undefined;   // caller falls back to the canonical CPU blend
    }
  }
  const packMs = performance.now() - packStarted;

  const result: CompositeResult = {
    type: "result",
    jobId: message.jobId,
    index: message.index,
    box: message.box,
    width,
    height,
    predBgr: predOut.buffer as ArrayBuffer,
    support: supportOut.buffer as ArrayBuffer,
    jawProtected: jawOut.buffer as ArrayBuffer,
    ...(preparedPrediction && roiSupport
      ? {
        preparedPrediction: preparedPrediction.buffer as ArrayBuffer,
        roiSupport: roiSupport.buffer as ArrayBuffer,
      }
      : {}),
    bitmap,
    renderMs: message.renderMs,
    compositorMs,
    packMs,
  };
  const transfer: Transferable[] = [
    result.predBgr, result.support, result.jawProtected,
  ];
  if (result.preparedPrediction) transfer.push(result.preparedPrediction);
  if (result.roiSupport) transfer.push(result.roiSupport);
  if (bitmap) transfer.push(bitmap);
  scope.postMessage(result, transfer);
};
