import type {
  CompositeJob, CompositeReady, CompositeResult,
} from "./composite-worker";
import type { RendererSpatialContract } from "./generated/runtime-tier-contract";
import type { RendererTemporalContract } from "./renderer-temporal";

export type CompositeJobInput = Omit<CompositeJob, "type" | "jobId">;

interface PendingComposite {
  resolve: (result: CompositeResult) => void;
  reject: (error: Error) => void;
}

/** Temporal finalization is ordered separately; it must not collapse this pool. */
export function resolvedCompositorWorkerCount(
  requested: number,
  _temporalEnabled = false,
): number {
  if (!Number.isSafeInteger(requested) || requested < 0) {
    throw new Error("compositor worker count must be a non-negative integer");
  }
  return requested;
}

/** Serialize stateful finalization in submission order while preparation stays parallel. */
export class OrderedCompositeFinalizer {
  private tail: Promise<void> = Promise.resolve();

  enqueue<T>(
    prepared: Promise<T>,
    finalize: (value: T) => void | Promise<void>,
  ): Promise<void> {
    const task = this.tail.then(async () => finalize(await prepared));
    this.tail = task;
    return task;
  }
}

/**
 * Fixed pool of QA9 compositor workers, created inside the pipeline worker.
 * Round-robins jobs so the per-frame CPU compositor runs across all spare cores
 * while the pipeline worker keeps the serial WebGPU renderer busy. Output is
 * byte-identical to the inline path; only placement changes.
 */
export class CompositePool {
  private readonly workers: Worker[] = [];
  private readonly pending = new Map<number, PendingComposite>();
  private failure?: Error;
  private nextJob = 0;
  private roundRobin = 0;

  private constructor(readonly size: number) {}

  static async create(
    size: number,
    support: Float32Array,
    hole: Float32Array,
    supportMultiplier?: Float32Array,
    rendererSpatialContract?: RendererSpatialContract,
    rendererTemporalContract?: RendererTemporalContract,
  ): Promise<CompositePool> {
    const pool = new CompositePool(size);
    const readies: Promise<void>[] = [];
    for (let i = 0; i < size; i += 1) {
      const worker = new Worker(
        new URL("./composite-worker.ts", import.meta.url),
        { type: "module", name: `serve320-composite-${i}` },
      );
      worker.onmessage = (event: MessageEvent<CompositeResult | CompositeReady>) => {
        const message = event.data;
        if (message.type === "result") {
          const pending = pool.pending.get(message.jobId);
          if (pending) {
            pool.pending.delete(message.jobId);
            pending.resolve(message);
          }
        }
      };
      const workerError = (event: ErrorEvent | MessageEvent) => {
        const detail = event instanceof ErrorEvent
          ? event.message
          : "message deserialization failed";
        pool.fail(new Error(`compositor worker ${i} failed: ${detail}`));
      };
      worker.addEventListener("error", workerError);
      worker.addEventListener("messageerror", workerError);
      readies.push(new Promise<void>((resolve, reject) => {
        const onReady = (event: MessageEvent<CompositeResult | CompositeReady>) => {
          if (event.data.type === "ready") {
            cleanup();
            resolve();
          }
        };
        const onStartupError = (event: ErrorEvent | MessageEvent) => {
          cleanup();
          const detail = event instanceof ErrorEvent
            ? event.message
            : "message deserialization failed";
          reject(new Error(`compositor worker ${i} failed to start: ${detail}`));
        };
        const cleanup = () => {
          worker.removeEventListener("message", onReady);
          worker.removeEventListener("error", onStartupError);
          worker.removeEventListener("messageerror", onStartupError);
        };
        worker.addEventListener("message", onReady);
        worker.addEventListener("error", onStartupError);
        worker.addEventListener("messageerror", onStartupError);
      }));
      // Copy the constant planes into each worker (shared read-only data).
      worker.postMessage(
        {
          type: "init",
          support: support.slice().buffer,
          hole: hole.slice().buffer,
          ...(supportMultiplier
            ? { supportMultiplier: supportMultiplier.slice().buffer }
            : {}),
          ...(rendererSpatialContract ? { rendererSpatialContract } : {}),
          ...(rendererTemporalContract ? { rendererTemporalContract } : {}),
        },
        [],
      );
      pool.workers.push(worker);
    }
    try {
      await Promise.all(readies);
      return pool;
    } catch (error) {
      pool.dispose();
      throw error;
    }
  }

  submit(job: CompositeJobInput, transfers: Transferable[]): Promise<CompositeResult> {
    if (this.failure) return Promise.reject(this.failure);
    if (this.workers.length === 0) {
      return Promise.reject(new Error("compositor pool is disposed"));
    }
    const jobId = this.nextJob;
    this.nextJob += 1;
    const worker = this.workers[this.roundRobin];
    this.roundRobin = (this.roundRobin + 1) % this.workers.length;
    return new Promise<CompositeResult>((resolve, reject) => {
      this.pending.set(jobId, { resolve, reject });
      try {
        worker.postMessage({ type: "job", jobId, ...job }, transfers);
      } catch (error) {
        this.pending.delete(jobId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  dispose(): void {
    const error = this.failure ?? new Error("compositor pool disposed");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    for (const worker of this.workers) worker.terminate();
    this.workers.length = 0;
  }

  private fail(error: Error): void {
    if (this.failure) return;
    this.failure = error;
    this.dispose();
  }
}
