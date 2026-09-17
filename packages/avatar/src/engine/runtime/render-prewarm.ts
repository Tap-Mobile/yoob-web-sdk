import type { MainToWorker } from "./protocol";

/** Measured point where ORT-Web graph capture reaches stable per-frame latency. */
export const RENDERER_STEADY_STATE_PASSES = 250;

/**
 * Only renderer work should yield the background warm-up. A microphone/VAD
 * cancel changes the response epoch but does not use the renderer, so stopping
 * warm-up there makes the first assistant reply pay the cold GPU ramp for no
 * latency benefit.
 */
export function interruptsRendererPrewarm(type: MainToWorker["type"]): boolean {
  return type === "prepare" || type === "prepare-chunk";
}

export function remainingRendererPrewarmPasses(completedPasses: number): number {
  const completed = Number.isFinite(completedPasses)
    ? Math.max(0, Math.floor(completedPasses))
    : 0;
  return Math.max(0, RENDERER_STEADY_STATE_PASSES - completed);
}
