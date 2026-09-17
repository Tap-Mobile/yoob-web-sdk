import type { RuntimeAssetConfig } from "../assets/runtime-store";
import type {
  RendererInputType, RendererPreferredLayout, RendererSpatialContract,
} from "./generated/runtime-tier-contract";
import type { NeuralMouthRenderStride } from "./render-cadence";
import type { RendererTemporalContract } from "./renderer-temporal";

export interface RuntimePressureTrace {
  type: "runtime-pressure";
  epoch: number;
  chunkIndex?: number;
  final: boolean;
  queueDepthAtEnqueue: number;
  geometryParallelAtEnqueue: number;
  geometryMs: number;
  rendererQueueWaitMs: number;
  renderMs: number;
  totalMs: number;
  compositorWorkers: number;
  compositorCapacity: number;
  maxCompositorInFlight: number;
}

export type MainToWorker =
  | {
      type: "initialize";
      assets: RuntimeAssetConfig;
      neuralMouthStride?: NeuralMouthRenderStride;
      rendererInputType?: RendererInputType;
      rendererPreferredLayout?: RendererPreferredLayout;
      rendererSpatialContract?: RendererSpatialContract;
      rendererTemporalContract?: RendererTemporalContract;
      compositorWorkers?: number;
    }
  | { type: "prepare"; epoch: number; pcm16le24k: ArrayBuffer }
  | {
      type: "prepare-chunk";
      epoch: number;
      chunkIndex: number;
      frameOffset: number;
      discardFrames: number;
      outputFrames: number;
      final: boolean;
      /** When set, overrides `final` for BiGRU silence-tail padding. */
      geometryFinal?: boolean;
      bootstrap?: boolean;
      pcm16le24k: ArrayBuffer;
    }
  | { type: "cancel"; epoch: number }
  /** A download grant renewed by a session heartbeat. Later downloads use it. */
  | { type: "grant"; downloadToken: string };

export type WorkerToMain =
  | {
      type: "status";
      message: string;
      level?: "info" | "success" | "warning";
      stage?: string;
      path?: string;
      progress?: number;
      loadedBytes?: number;
      totalBytes?: number;
      cached?: boolean;
      elapsedMs?: number;
    }
  | { type: "initialized"; nIdle: number }
  | { type: "geometry-ready"; epoch: number; frameCount: number; audioSamples: number }
  | {
      type: "chunk-geometry-ready";
      epoch: number;
      chunkIndex: number;
      frameOffset: number;
      frameCount: number;
      final: boolean;
    }
  | {
      type: "frame";
      epoch: number;
      index: number;
      box: [number, number, number, number];
      width: number;
      height: number;
      predBgr: ArrayBuffer;
      support: ArrayBuffer;
      jawProtected: ArrayBuffer;
      /**
       * Straight-alpha mouth RGBA packed by the compositor worker. When
       * present the coordinator lets the GPU do the alpha blend via drawImage
       * instead of the 4.4 ms/frame main-thread canonicalBlendRgba loop.
       * Absent on the inline (no-pool) path, which keeps the CPU blend.
       */
      bitmap?: ImageBitmap;
      renderMs: number;
    }
  | {
      type: "render-complete";
      epoch: number;
      frameCount: number;
      meanRenderMs: number;
      featherMs?: number;
      geometryMs?: number;
      rendererMsMean?: number;
      neuralFrames?: number;
      compositorMsMean?: number;
    }
  | {
      type: "chunk-render-complete";
      epoch: number;
      chunkIndex: number;
      frameOffset: number;
      frameCount: number;
      final: boolean;
      meanRenderMs: number;
    }
  | RuntimePressureTrace
  | { type: "error"; epoch?: number; message: string; stack?: string };
