import { createSession, ort } from "./ort-runtime";
import { createFloat16InputStaging, type Float16InputStaging } from "./renderer-input";
import {
  FULL320_RENDERER_SPATIAL_CONTRACT,
  rendererInputLength,
  rendererInputShape,
  rendererOutputLength,
  rendererOutputShape,
  resolveRendererSpatialContract,
} from "../runtime/renderer-contract";
import type {
  RendererInputType, RendererPreferredLayout, RendererSpatialContract,
} from "../runtime/generated/runtime-tier-contract";

/**
 * WebGPU graph capture replays a pre-recorded command buffer instead of
 * re-encoding every dispatch per frame. On the frozen bench harness this is
 * the largest renderer win available — 36.7 ms vs 68.9 ms for the default
 * session on Apple metal-3 — with byte-identical output (`u8Mismatches: 0`
 * across every variant in `bench.html?mode=renderer`).
 *
 * It requires static shapes (declared once by the selected immutable runtime
 * contract) plus GPU-resident input and output, so frame data is staged through
 * persistent buffers and read back explicitly.
 */
interface GraphCaptureContext {
  device: GPUDevice;
  inputBuffer: GPUBuffer;
  outputBuffer: GPUBuffer;
  readbackBuffer: GPUBuffer;
  inputTensor: ort.Tensor;
  outputTensor: ort.Tensor;
  mapRead: number;
}

function gpuConstants() {
  const scope = globalThis as unknown as {
    GPUBufferUsage?: {
      COPY_DST: number; COPY_SRC: number; MAP_READ: number; STORAGE: number;
    };
    GPUMapMode?: { READ: number };
  };
  if (!scope.GPUBufferUsage || !scope.GPUMapMode) return undefined;
  return { usage: scope.GPUBufferUsage, mapMode: scope.GPUMapMode };
}

export class RendererRuntime {
  private inputTensor?: ort.Tensor;
  private inputBuffer?: Float32Array;
  private readonly float16Input?: Float16InputStaging;
  readonly spatialContract: RendererSpatialContract;
  private readonly inputShape: number[];
  private readonly outputShape: number[];
  private readonly inputLength: number;
  private readonly outputLength: number;
  private readonly outputBytes: number;
  /** Tail of the graph-capture serialization chain (see runCaptured). */
  private captureQueue: Promise<void> = Promise.resolve();

  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly rendererInputType: RendererInputType,
    readonly activeLayout: RendererPreferredLayout,
    spatialContract: RendererSpatialContract,
    private readonly capture?: GraphCaptureContext,
  ) {
    this.spatialContract = resolveRendererSpatialContract(spatialContract);
    this.inputShape = rendererInputShape(this.spatialContract);
    this.outputShape = rendererOutputShape(this.spatialContract);
    this.inputLength = rendererInputLength(this.spatialContract);
    this.outputLength = rendererOutputLength(this.spatialContract);
    this.outputBytes = this.outputLength * Float32Array.BYTES_PER_ELEMENT;
    if (rendererInputType === "float16") {
      this.float16Input = createFloat16InputStaging(this.inputLength);
    }
  }

  /** True when the fast replay path is active (diagnostics only). */
  get graphCaptureEnabled(): boolean {
    return this.capture !== undefined;
  }

  static async load(
    model: string | Uint8Array,
    onFallback?: (message: string) => void,
    onStatus?: (message: string) => void,
    rendererInputType: RendererInputType = "float32",
    rendererPreferredLayout: RendererPreferredLayout = "NCHW",
    rendererSpatialContract: RendererSpatialContract = FULL320_RENDERER_SPATIAL_CONTRACT,
  ) {
    const spatialContract = resolveRendererSpatialContract(rendererSpatialContract);
    const layouts: RendererPreferredLayout[] = rendererPreferredLayout === "NHWC"
      ? ["NHWC", "NCHW"] : ["NCHW"];
    for (const layout of layouts) {
      const captured = await RendererRuntime.tryGraphCapture(
        model, rendererInputType, layout, spatialContract, onStatus,
      );
      if (captured) return captured;
      if (layout === "NHWC") {
        onStatus?.("Renderer NHWC capture rejected; retrying safe NCHW capture");
      }
    }
    return new RendererRuntime(await createSession(
      model, ["webgpu", "wasm"], "Serve320 renderer", onFallback,
    ), rendererInputType, "NCHW", spatialContract);
  }

  /**
   * Best-effort graph-capture session. Any failure here is non-fatal: the
   * caller falls back to the portable session, so a driver that rejects
   * capture costs a slower renderer rather than a broken runtime.
   */
  private static async tryGraphCapture(
    model: string | Uint8Array,
    rendererInputType: RendererInputType,
    rendererPreferredLayout: RendererPreferredLayout,
    spatialContract: RendererSpatialContract,
    onStatus?: (message: string) => void,
  ): Promise<RendererRuntime | undefined> {
    const constants = gpuConstants();
    if (!constants || typeof ort.Tensor.fromGpuBuffer !== "function") return undefined;
    let session: ort.InferenceSession | undefined;
    let inputBuffer: GPUBuffer | undefined;
    let outputBuffer: GPUBuffer | undefined;
    let readbackBuffer: GPUBuffer | undefined;
    try {
      session = await createSession(
        model, ["webgpu"], "Serve320 renderer · graph capture", undefined,
        { preferredOutputLocation: "gpu-buffer", enableGraphCapture: true },
        rendererPreferredLayout,
      );
      const device = await ort.env.webgpu.device as GPUDevice;
      if (!device) throw new Error("no WebGPU device");
      const inputLength = rendererInputLength(spatialContract);
      const outputLength = rendererOutputLength(spatialContract);
      const outputBytes = outputLength * Float32Array.BYTES_PER_ELEMENT;
      const inputShape = rendererInputShape(spatialContract);
      const outputShape = rendererOutputShape(spatialContract);
      inputBuffer = device.createBuffer({
        size: inputLength * (rendererInputType === "float16"
          ? Uint16Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT),
        usage: constants.usage.COPY_DST | constants.usage.STORAGE,
      });
      outputBuffer = device.createBuffer({
        size: outputBytes,
        usage: constants.usage.COPY_SRC | constants.usage.STORAGE,
      });
      readbackBuffer = device.createBuffer({
        size: outputBytes,
        usage: constants.usage.COPY_DST | constants.usage.MAP_READ,
      });
      (inputBuffer as unknown as { __ortPin?: boolean }).__ortPin = true;
      (outputBuffer as unknown as { __ortPin?: boolean }).__ortPin = true;
      const runtime = new RendererRuntime(
        session, rendererInputType, rendererPreferredLayout, spatialContract, {
        device,
        inputBuffer,
        outputBuffer,
        readbackBuffer,
        // ORT 1.27's WebGPU EP is native C++ via emdawnwebgpu: a GPUBuffer handle
      // IS the WASM heap address of the C++ WGPUBufferImpl, and `run()`'s
      // `finally` unregisters caller-owned external buffers on EVERY run. The
      // refcount therefore reaches 0 after run 1, the impl is freed back into
      // the shared heap, and the captured graph's binding points at a recycled
      // address — surfacing as "createBindGroup ... 'buffer' ... Required
      // member is undefined" on replay. Whether it detonates depends on
      // allocator luck, which is why an unrelated smaller geometry model
      // flipped it deterministically.
      //
      // `__ortPin` is honoured by a patch to onnxruntime-web's
      // webgpuUnregisterBuffer (see node_modules/.../ort.webgpu.bundle.min.mjs,
      // *.orig kept alongside): release-on-zero is skipped for pinned buffers,
      // so the handle stays valid for the session's lifetime — the same
      // lifetime ORT already grants its own EP-owned buffers.
      inputTensor: ort.Tensor.fromGpuBuffer(inputBuffer, {
          dataType: rendererInputType, dims: inputShape,
        }),
        outputTensor: ort.Tensor.fromGpuBuffer(outputBuffer, {
          dataType: "float32", dims: outputShape,
        }),
        mapRead: constants.mapMode.READ,
      });
      // Prove the capture across THREE runs before adopting it.
      //
      // ORT does one ordinary run, records the graph on the next, and only then
      // replays. A single proving run therefore returns before the captured
      // binding is ever exercised, so a broken replay escaped this try/catch
      // and detonated later during warmup instead of falling back to the
      // portable session. Three runs covers regular -> capture -> replay.
      const probe = new Float32Array(inputLength);
      for (let attempt = 0; attempt < 3; attempt += 1) await runtime.run(probe);
      onStatus?.(
        `Renderer graph capture active · ${rendererInputType} input · `
        + `${rendererPreferredLayout} layout`,
      );
      return runtime;
    } catch (error) {
      inputBuffer?.destroy();
      outputBuffer?.destroy();
      readbackBuffer?.destroy();
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (session as any)?.release?.();
      } catch { /* release is best-effort during fallback */ }
      onStatus?.(
        `Renderer ${rendererPreferredLayout} graph capture unavailable · `
        + (error instanceof Error ? error.message : String(error)),
      );
      return undefined;
    }
  }

  /**
   * Run one renderer step. When `output` is provided, results are copied into
   * it (no per-frame Float32Array allocation). Input may be retained by the
   * session for the duration of the call — do not mutate it until the promise
   * resolves.
   */
  async run(input: Float32Array, output?: Float32Array): Promise<Float32Array> {
    if (input.length !== this.inputLength) throw new Error(`bad renderer input ${input.length}`);
    if (output && output.length !== this.outputLength) {
      throw new Error(`bad renderer output buffer ${output.length}`);
    }
    if (this.capture) return this.runCaptured(input, output);
    if (this.rendererInputType === "float16") {
      const bits = this.float16Input!.write(input);
      this.inputTensor ??= new ort.Tensor("float16", bits, this.inputShape);
    } else if (!this.inputTensor || this.inputBuffer !== input) {
      this.inputBuffer = input;
      this.inputTensor = new ort.Tensor("float32", input, this.inputShape);
    }
    const result = await this.session.run({ x: this.inputTensor });
    const tensor = result.y;
    if (!tensor || tensor.dims.join(",") !== this.outputShape.join(",")) {
      throw new Error(`bad renderer output ${tensor?.dims}`);
    }
    const data = tensor.data as Float32Array;
    if (output) {
      output.set(data);
      tensor.dispose();
      return output;
    }
    const copy = data.slice();
    tensor.dispose();
    return copy;
  }

  private async runCaptured(
    input: Float32Array, output?: Float32Array,
  ): Promise<Float32Array> {
    // The pipeline deliberately overlaps renderer runs (it dispatches the next
    // neural frame while the current one is still in flight) but the capture
    // context owns a single input/output/readback buffer set. Two concurrent
    // runs corrupt each other's bind groups — surfacing as
    // "Failed to read the 'buffer' property from 'GPUBufferBinding'" — and a
    // second mapAsync on an already-mapped buffer is invalid. Serialize.
    const previous = this.captureQueue;
    let release!: () => void;
    this.captureQueue = new Promise<void>((resolve) => { release = resolve; });
    await previous;
    try {
      return await this.runCapturedExclusive(input, output);
    } finally {
      release();
    }
  }

  private async runCapturedExclusive(
    input: Float32Array, output?: Float32Array,
  ): Promise<Float32Array> {
    const capture = this.capture!;
    const upload = this.rendererInputType === "float16"
      ? this.float16Input!.write(input) : input;
    capture.device.queue.writeBuffer(capture.inputBuffer, 0, upload as Float32Array<ArrayBuffer>);
    await this.session.run(
      { x: capture.inputTensor }, { y: capture.outputTensor },
    );
    const encoder = capture.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      capture.outputBuffer, 0, capture.readbackBuffer, 0, this.outputBytes,
    );
    capture.device.queue.submit([encoder.finish()]);
    await capture.readbackBuffer.mapAsync(capture.mapRead);
    const mapped = new Float32Array(capture.readbackBuffer.getMappedRange());
    const result = output ?? new Float32Array(this.outputLength);
    result.set(mapped);
    capture.readbackBuffer.unmap();
    return result;
  }

  async warmup(input?: Float32Array): Promise<void> {
    await this.run(input ?? new Float32Array(this.inputLength));
  }

  /**
   * Grind the real capture session past its warm-up cliff.
   *
   * Measured on this stack: the first ~250 renderer passes cost 34.2–44.3 ms
   * each and the whole pipeline delivers only 8–13 fps, while every run after
   * that sustains 30.1–30.6 fps at 23.9–24.2 ms. That is ORT-Web WebGPU
   * pipeline compilation plus the GPU clock ramp, and it lasts roughly one
   * conversational turn — so the FIRST reply is the one users see stutter,
   * which reads as "blurry, steppy lips and fewer fps" against iOS. iOS has no
   * counterpart: CoreML ships precompiled .mlmodelc to the ANE.
   *
   * It is not paid before "ready" because that would just move ~10 s of
   * blocking wait to the loading screen. It runs after, and `shouldStop` lets
   * real work cancel it immediately — a half-warmed session is strictly better
   * than a delayed first frame.
   */
  async prewarmSteadyState(
    input: Float32Array,
    iterations: number,
    shouldStop: () => boolean,
    onProgress?: (done: number, total: number) => void,
  ): Promise<number> {
    let done = 0;
    for (let index = 0; index < iterations; index += 1) {
      if (shouldStop()) break;
      await this.run(input);
      done += 1;
      // Yield so a queued prepare/render message is seen promptly rather than
      // after the whole loop; without this the cancel check never fires.
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (onProgress && done % 50 === 0) onProgress(done, iterations);
    }
    return done;
  }
}
