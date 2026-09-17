const PCM_SAMPLE_RATE = 24_000;

/** Initial queued PCM required before streaming playback may start. */
export const STREAMING_PCM_START_RESERVE_MS = 400;
/** Hard latency ceiling for reserve adaptation across turns. */
export const STREAMING_PCM_MAX_START_RESERVE_MS = 560;
/** One 25-fps frame per adaptation step. */
export const STREAMING_PCM_RESERVE_STEP_MS = 40;

/**
 * The reserve remains an explicit field experiment until matched authenticated
 * conversations prove that the underrun reduction outweighs its startup cost.
 */
export function streamingPcmReserveEnabled(search: string): boolean {
  return new URLSearchParams(search).get("pcmreserve") === "1";
}

export function pcmSamplesForMs(milliseconds: number): number {
  return Math.round(milliseconds * PCM_SAMPLE_RATE / 1_000);
}

/**
 * Increase the next turn's start reserve only after a measured audio underrun.
 * A clean turn relaxes by one frame, never below the 400 ms floor. Renderer
 * pressure is deliberately excluded: mouth-frame readiness has its own gate.
 */
export function nextStreamingPcmStartReserveSamples(
  currentSamples: number,
  playbackUnderruns: number,
): number {
  const minimum = pcmSamplesForMs(STREAMING_PCM_START_RESERVE_MS);
  const maximum = pcmSamplesForMs(STREAMING_PCM_MAX_START_RESERVE_MS);
  const step = pcmSamplesForMs(STREAMING_PCM_RESERVE_STEP_MS);
  const bounded = Math.min(maximum, Math.max(minimum, Math.round(currentSamples)));
  return playbackUnderruns > 0
    ? Math.min(maximum, bounded + step)
    : Math.max(minimum, bounded - step);
}

/**
 * Playback owns neither clock until both independent supplies are ready:
 * contiguous rendered mouths and queued PCM. A finalized short reply may use
 * all of its available PCM instead of waiting for an impossible reserve.
 */
export function streamingAudioStartReady(
  queuedSamples: number,
  startReserveSamples: number,
  renderedPrefix: number,
  renderStartTarget: number,
  finalReceived = false,
): boolean {
  if (queuedSamples <= 0 || renderStartTarget <= 0) return false;
  const pcmReady = finalReceived || queuedSamples >= startReserveSamples;
  return pcmReady && renderedPrefix >= renderStartTarget;
}
