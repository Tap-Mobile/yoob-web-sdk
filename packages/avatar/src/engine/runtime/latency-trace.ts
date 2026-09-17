/**
 * Lightweight latency chain for the conversational avatar path.
 * Marks are performance.now() relative to the first mark in the session.
 *
 * Primary commercial metric: pcm → first presented speaking frame.
 * Secondary (Annie-style presence): user speech → first listen reaction.
 */
export type LatencyMarkName =
  | "session_start"
  | "mic_ready"
  | "client_speech_started"
  | "client_speech_stopped"
  | "user_speech_started"
  | "user_speech_stopped"
  | "first_listen_reaction"
  | "asr_final"
  | "response_created"
  | "first_assistant_text"
  | "first_assistant_pcm"
  | "first_render_bucket"
  | "first_composited_frame"
  | "audio_start"
  | "first_presented_frame"
  | "first_render_underrun"
  | "first_playback_underrun"
  | "audio_final"
  | "playback_end";

const ORDER: LatencyMarkName[] = [
  "session_start",
  "mic_ready",
  "client_speech_started",
  "client_speech_stopped",
  "user_speech_started",
  "user_speech_stopped",
  "first_listen_reaction",
  "asr_final",
  "response_created",
  "first_assistant_text",
  "first_assistant_pcm",
  "first_render_bucket",
  "first_composited_frame",
  "audio_start",
  "first_presented_frame",
  "first_render_underrun",
  "first_playback_underrun",
  "audio_final",
  "playback_end",
];

const TURN_MARKS: LatencyMarkName[] = [
  "client_speech_started",
  "client_speech_stopped",
  "user_speech_started",
  "user_speech_stopped",
  "first_listen_reaction",
  "asr_final",
  "response_created",
  "first_assistant_text",
  "first_assistant_pcm",
  "first_render_bucket",
  "first_composited_frame",
  "audio_start",
  "first_presented_frame",
  "first_render_underrun",
  "first_playback_underrun",
  "audio_final",
  "playback_end",
];

export interface LatencyReport {
  marks: Partial<Record<LatencyMarkName, number>>;
  pcmToFirstFrameMs?: number;
  pcmToAudioMs?: number;
  speechStopToPcmMs?: number;
  speechStopToAudioMs?: number;
  speechToListenReactionMs?: number;
  /** Renderer pressure observations; audio is intentionally never paused. */
  underrunCount: number;
  playbackUnderrunCount: number;
  playbackUnderrunMs: number;
  maxPlaybackUnderrunMs: number;
  streamingStartTargetMs?: number;
  minStreamingBufferMs?: number;
  peakStreamingBufferMs: number;
  summary: string;
}

export class LatencyTrace {
  private readonly marks = new Map<LatencyMarkName, number>();
  private origin = 0;
  private underrunCount = 0;
  private playbackUnderrunCount = 0;
  private playbackUnderrunMs = 0;
  private maxPlaybackUnderrunMs = 0;
  private streamingStartTargetMs?: number;
  private minStreamingBufferMs?: number;
  private peakStreamingBufferMs = 0;

  reset(): void {
    this.marks.clear();
    this.origin = 0;
    this.resetTurnMetrics();
  }

  /**
   * Drop per-turn marks. Also clears user-speech markers so the next turn's
   * speech→listen / pcm→frame deltas are not anchored to an earlier utterance.
   */
  beginTurn(): void {
    for (const name of TURN_MARKS) this.marks.delete(name);
    this.resetTurnMetrics();
  }

  private resetTurnMetrics(): void {
    this.underrunCount = 0;
    this.playbackUnderrunCount = 0;
    this.playbackUnderrunMs = 0;
    this.maxPlaybackUnderrunMs = 0;
    this.streamingStartTargetMs = undefined;
    this.minStreamingBufferMs = undefined;
    this.peakStreamingBufferMs = 0;
  }

  mark(name: LatencyMarkName, at = performance.now()): void {
    if (this.marks.has(name)) return;
    if (this.marks.size === 0) this.origin = at;
    this.marks.set(name, at);
    try {
      performance.mark(`serve320:${name}`);
    } catch {
      // performance.mark is optional in non-browser test hosts
    }
  }

  /** Count a render underrun (audio paused waiting for frames). */
  noteUnderrun(at = performance.now()): void {
    this.underrunCount += 1;
    this.mark("first_render_underrun", at);
  }

  has(name: LatencyMarkName): boolean {
    return this.marks.has(name);
  }

  getUnderrunCount(): number {
    return this.underrunCount;
  }

  notePlaybackBuffer(bufferedSamples: number, started: boolean, final: boolean): void {
    const bufferedMs = Math.max(0, bufferedSamples) * 1_000 / 24_000;
    this.peakStreamingBufferMs = Math.max(this.peakStreamingBufferMs, bufferedMs);
    if (started && !final) {
      this.minStreamingBufferMs = this.minStreamingBufferMs === undefined
        ? bufferedMs
        : Math.min(this.minStreamingBufferMs, bufferedMs);
    }
  }

  notePlaybackUnderrun(durationSamples: number, at = performance.now()): void {
    const durationMs = Math.max(0, durationSamples) * 1_000 / 24_000;
    if (durationMs <= 0) return;
    this.playbackUnderrunCount += 1;
    this.playbackUnderrunMs += durationMs;
    this.maxPlaybackUnderrunMs = Math.max(this.maxPlaybackUnderrunMs, durationMs);
    this.mark("first_playback_underrun", at);
  }

  noteStreamingStartTargetSamples(samples: number): void {
    this.streamingStartTargetMs = Math.max(0, samples) * 1_000 / 24_000;
  }

  /** Milliseconds from origin (first mark) to this mark, or undefined. */
  at(name: LatencyMarkName): number | undefined {
    const value = this.marks.get(name);
    return value === undefined ? undefined : value - this.origin;
  }

  /** Delta between two marks in ms, or undefined if either is missing. */
  delta(from: LatencyMarkName, to: LatencyMarkName): number | undefined {
    const a = this.marks.get(from);
    const b = this.marks.get(to);
    if (a === undefined || b === undefined) return undefined;
    return b - a;
  }

  /** pcm → presented speaking frame (primary commercial metric). */
  pcmToFirstFrameMs(): number | undefined {
    return this.delta("first_assistant_pcm", "first_presented_frame");
  }

  pcmToAudioMs(): number | undefined {
    return this.delta("first_assistant_pcm", "audio_start");
  }

  /** Server VAD boundary → first assistant PCM received by the browser. */
  speechStopToPcmMs(): number | undefined {
    return this.delta("user_speech_stopped", "first_assistant_pcm");
  }

  /** Server VAD boundary → synchronized audio playback start. */
  speechStopToAudioMs(): number | undefined {
    return this.delta("user_speech_stopped", "audio_start");
  }

  /** Server VAD boundary → first presented speaking frame. */
  speechStopToFirstFrameMs(): number | undefined {
    return this.delta("user_speech_stopped", "first_presented_frame");
  }

  /** LiveKit client active-speaker stop → authoritative agent VAD boundary. */
  clientStopToServerStopMs(): number | undefined {
    return this.delta("client_speech_stopped", "user_speech_stopped");
  }

  /** Client active-speaker stop → first assistant PCM received by the browser. */
  clientStopToPcmMs(): number | undefined {
    return this.delta("client_speech_stopped", "first_assistant_pcm");
  }

  /** Client active-speaker stop → synchronized audio playback start. */
  clientStopToAudioMs(): number | undefined {
    return this.delta("client_speech_stopped", "audio_start");
  }

  /** Client active-speaker stop → first presented speaking frame. */
  clientStopToFirstFrameMs(): number | undefined {
    return this.delta("client_speech_stopped", "first_presented_frame");
  }

  /** Earliest client/agent speech start → first listen reaction (presence metric). */
  speechToListenReactionMs(): number | undefined {
    const start = this.marks.has("client_speech_started")
      ? "client_speech_started"
      : "user_speech_started";
    return this.delta(start, "first_listen_reaction");
  }

  summary(): string {
    const parts: string[] = [];
    for (const name of ORDER) {
      const ms = this.at(name);
      if (ms !== undefined) parts.push(`${name}=${ms.toFixed(0)}ms`);
    }
    const pcmFrame = this.pcmToFirstFrameMs();
    if (pcmFrame !== undefined) parts.push(`pcm→frame=${pcmFrame.toFixed(0)}ms`);
    const pcmAudio = this.pcmToAudioMs();
    if (pcmAudio !== undefined) parts.push(`pcm→audio=${pcmAudio.toFixed(0)}ms`);
    const stopPcm = this.speechStopToPcmMs();
    if (stopPcm !== undefined) parts.push(`speech-stop→pcm=${stopPcm.toFixed(0)}ms`);
    const stopAudio = this.speechStopToAudioMs();
    if (stopAudio !== undefined) parts.push(`speech-stop→audio=${stopAudio.toFixed(0)}ms`);
    const stopFrame = this.speechStopToFirstFrameMs();
    if (stopFrame !== undefined) parts.push(`speech-stop→frame=${stopFrame.toFixed(0)}ms`);
    const clientServer = this.clientStopToServerStopMs();
    if (clientServer !== undefined) {
      parts.push(`client-stop→agent-stop=${clientServer.toFixed(0)}ms`);
    }
    const clientPcm = this.clientStopToPcmMs();
    if (clientPcm !== undefined) parts.push(`client-stop→pcm=${clientPcm.toFixed(0)}ms`);
    const clientAudio = this.clientStopToAudioMs();
    if (clientAudio !== undefined) parts.push(`client-stop→audio=${clientAudio.toFixed(0)}ms`);
    const clientFrame = this.clientStopToFirstFrameMs();
    if (clientFrame !== undefined) parts.push(`client-stop→frame=${clientFrame.toFixed(0)}ms`);
    const listen = this.speechToListenReactionMs();
    if (listen !== undefined) parts.push(`speech→listen=${listen.toFixed(0)}ms`);
    if (this.underrunCount > 0) parts.push(`underruns=${this.underrunCount}`);
    parts.push(
      `audio-underruns=${this.playbackUnderrunCount}/${this.playbackUnderrunMs.toFixed(0)}ms`,
    );
    if (this.streamingStartTargetMs !== undefined) {
      parts.push(`pcm-start-target=${this.streamingStartTargetMs.toFixed(0)}ms`);
    }
    if (this.minStreamingBufferMs !== undefined) {
      parts.push(
        `pcm-buffer=${this.minStreamingBufferMs.toFixed(0)}-${this.peakStreamingBufferMs.toFixed(0)}ms`,
      );
    }
    return parts.join(" · ");
  }

  snapshot(): Partial<Record<LatencyMarkName, number>> {
    const out: Partial<Record<LatencyMarkName, number>> = {};
    for (const name of ORDER) {
      const ms = this.at(name);
      if (ms !== undefined) out[name] = ms;
    }
    return out;
  }

  /** Structured report for SDK / metrics consumers. */
  report(): LatencyReport {
    return {
      marks: this.snapshot(),
      pcmToFirstFrameMs: this.pcmToFirstFrameMs(),
      pcmToAudioMs: this.pcmToAudioMs(),
      speechStopToPcmMs: this.speechStopToPcmMs(),
      speechStopToAudioMs: this.speechStopToAudioMs(),
      speechToListenReactionMs: this.speechToListenReactionMs(),
      underrunCount: this.underrunCount,
      playbackUnderrunCount: this.playbackUnderrunCount,
      playbackUnderrunMs: this.playbackUnderrunMs,
      maxPlaybackUnderrunMs: this.maxPlaybackUnderrunMs,
      streamingStartTargetMs: this.streamingStartTargetMs,
      minStreamingBufferMs: this.minStreamingBufferMs,
      peakStreamingBufferMs: this.peakStreamingBufferMs,
      summary: this.summary(),
    };
  }
}
