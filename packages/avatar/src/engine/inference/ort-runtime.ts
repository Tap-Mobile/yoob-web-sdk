import * as ort from "onnxruntime-web/webgpu";
import type { RendererPreferredLayout } from "../runtime/generated/runtime-tier-contract";

let configured = false;
let wasmUrl: string | undefined;

/** Where the ONNX Runtime WebAssembly binary is loaded from. Set before the first session opens. */
export function setOrtWasmUrl(url: string): void {
  wasmUrl = url;
}

export function configureOrt(): void {
  if (configured) return;
  configured = true;
  if (wasmUrl) ort.env.wasm.wasmPaths = { wasm: wasmUrl };
  // This runtime already runs inside a dedicated pipeline worker. ORT's
  // pthread bootstrap can deadlock when it tries to create nested workers from
  // a Vite-bundled module worker, leaving InferenceSession.create unresolved.
  // A single WASM thread avoids that bootstrap; WebGPU still accelerates the
  // renderer.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.initTimeout = 30_000;
  ort.env.wasm.simd = true;
  ort.env.webgpu.powerPreference = "high-performance";
}

export type WebExecutionProvider = "webgpu" | "wasm";

function openSession(
  model: string | Uint8Array,
  options: ort.InferenceSession.SessionOptions,
): Promise<ort.InferenceSession> {
  return typeof model === "string"
    ? ort.InferenceSession.create(model, options)
    : ort.InferenceSession.create(model, options);
}

export async function createSession(
  model: string | Uint8Array,
  executionProviders: WebExecutionProvider[],
  label: string,
  onFallback?: (message: string) => void,
  /**
   * Extra WebGPU-only session options (e.g. `enableGraphCapture` plus a
   * `gpu-buffer` output location for the renderer's replay path). Never
   * applied to the WASM fallback, which cannot honour GPU-resident tensors.
   */
  webgpuOptions?: ort.InferenceSession.SessionOptions,
  /** Optional internal WebGPU convolution layout. NCHW preserves ORT defaults. */
  webgpuPreferredLayout?: RendererPreferredLayout,
): Promise<ort.InferenceSession> {
  configureOrt();
  if (!executionProviders.includes("webgpu")) {
    return openSession(model, {
      executionProviders,
      graphOptimizationLevel: "all",
      executionMode: "sequential",
    });
  }
  try {
    const providers: ort.InferenceSession.ExecutionProviderConfig[] =
      webgpuPreferredLayout === "NHWC"
        ? executionProviders.map((provider) => provider === "webgpu"
          ? { name: "webgpu", preferredLayout: "NHWC" }
          : provider)
        : executionProviders;
    return await openSession(model, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      executionMode: "sequential",
      ...webgpuOptions,
    });
  } catch (webgpuError) {
    // A caller that asked for graph capture owns its own fallback; do not
    // hand it a WASM session that cannot replay a captured graph.
    if (webgpuOptions?.enableGraphCapture) throw webgpuError;
    const message = `${label}: WebGPU open failed; falling back to WASM · `
      + `${webgpuError instanceof Error ? webgpuError.message : String(webgpuError)}`;
    console.warn(message, webgpuError);
    onFallback?.(message);
    return openSession(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      executionMode: "sequential",
    });
  }
}

export { ort };
