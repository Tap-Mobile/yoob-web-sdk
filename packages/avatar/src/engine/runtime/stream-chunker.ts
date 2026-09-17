const SAMPLE_RATE = 24_000;
const FPS = 25;
export const STREAM_SAMPLES_PER_FRAME = SAMPLE_RATE / FPS;
// 600 ms payload + 1.0 s future context. Qwen generates ahead of the playback
// clock, so this keeps the bidirectional geometry quality contract without the
// old 1.2 s payload's visible first-word/next-sentence pacing gap.
export const STREAM_CHUNK_FRAMES = 15;
export const STREAM_CONTEXT_FRAMES = 25;
/** Safe first-turn bucket while the renderer has no measured history. */
export const STREAM_BOOTSTRAP_FRAMES = 4;
/** Fast bucket selected after this browser proves it renders comfortably in real time. */
export const STREAM_FAST_BOOTSTRAP_FRAMES = 3;
const STREAM_FAST_BOOTSTRAP_RENDER_MS = 30;

/**
 * Reuse the previous turn's renderer timing to trim the lip-sync bootstrap.
 * Unknown or slower devices stay at the proven 4-frame/160 ms bucket. A
 * renderer averaging at most 30 ms/frame uses 3 frames/120 ms on later turns.
 */
export function streamingBootstrapFrameCount(previousMeanRenderMs: number): number {
  return Number.isFinite(previousMeanRenderMs)
      && previousMeanRenderMs > 0
      && previousMeanRenderMs <= STREAM_FAST_BOOTSTRAP_RENDER_MS
    ? STREAM_FAST_BOOTSTRAP_FRAMES
    : STREAM_BOOTSTRAP_FRAMES;
}

export type StreamingWindowMode = "sliding" | "prefix";

export interface StreamingPcmChunkerOptions {
  chunkFrames?: number;
  contextFrames?: number;
  /** When > 0, emit a silence-tailed bootstrap before the first full window. */
  bootstrapFrames?: number;
  mode?: StreamingWindowMode;
}

export interface StreamingRenderChunk {
  modelPcm: Int16Array;
  playbackPcm: Int16Array;
  discardFrames: number;
  outputFrames: number;
  frameOffset: number;
  /** Stream is complete after this chunk. */
  final: boolean;
  /**
   * Pad the accepted BiGRU silence tail after modelPcm.
   * True for bootstrap mid-stream (future speech unknown) and for true finals.
   */
  geometryFinal: boolean;
  /** First early bucket before 30+25 sliding windows are ready. */
  bootstrap: boolean;
}

function concatenate(left: Int16Array, right: Int16Array): Int16Array<ArrayBuffer> {
  const joined = new Int16Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

/**
 * Produces render buckets for the bidirectional geometry model.
 *
 * Steady state: 0.6 s payload + 1.0 s future context (sliding or prefix history).
 * Optional bootstrap: once `bootstrapFrames` of real PCM exist and a full
 * payload+context window is not yet available, emit those frames immediately
 * with geometryFinal=true so the model sees the accepted n+25 silence tail.
 */
export class StreamingPcmChunker {
  private pending: Int16Array<ArrayBufferLike> = new Int16Array();
  private past: Int16Array<ArrayBufferLike> = new Int16Array();
  private prefix: Int16Array<ArrayBufferLike> = new Int16Array();
  private nextFrameOffset = 0;
  private ended = false;
  private bootstrapEmitted = false;
  readonly chunkFrames: number;
  readonly contextFrames: number;
  readonly bootstrapFrames: number;
  readonly mode: StreamingWindowMode;

  constructor(options: StreamingPcmChunkerOptions = {}) {
    this.chunkFrames = options.chunkFrames ?? STREAM_CHUNK_FRAMES;
    this.contextFrames = options.contextFrames ?? STREAM_CONTEXT_FRAMES;
    this.bootstrapFrames = options.bootstrapFrames ?? 0;
    this.mode = options.mode ?? "sliding";
    if (!Number.isSafeInteger(this.chunkFrames) || this.chunkFrames < 1) {
      throw new Error(`invalid streaming chunk frames ${this.chunkFrames}`);
    }
    if (!Number.isSafeInteger(this.contextFrames) || this.contextFrames < 0) {
      throw new Error(`invalid streaming context frames ${this.contextFrames}`);
    }
    if (!Number.isSafeInteger(this.bootstrapFrames) || this.bootstrapFrames < 0) {
      throw new Error(`invalid streaming bootstrap frames ${this.bootstrapFrames}`);
    }
    if (this.bootstrapFrames > 0 && this.bootstrapFrames >= this.chunkFrames) {
      throw new Error(
        `bootstrap frames ${this.bootstrapFrames} must be < chunk frames ${this.chunkFrames}`,
      );
    }
  }

  push(pcm: Int16Array, final = false): StreamingRenderChunk[] {
    if (this.ended) throw new Error("streaming PCM was already finalized");
    if (pcm.length > 0) this.pending = concatenate(this.pending, pcm);
    const output: StreamingRenderChunk[] = [];
    const payloadSamples = this.chunkFrames * STREAM_SAMPLES_PER_FRAME;
    const contextSamples = this.contextFrames * STREAM_SAMPLES_PER_FRAME;
    const readySamples = payloadSamples + contextSamples;
    const bootstrapSamples = this.bootstrapFrames * STREAM_SAMPLES_PER_FRAME;

    // Early first bucket: real audio only; geometry gets silence tail via geometryFinal.
    if (
      !this.bootstrapEmitted
      && this.bootstrapFrames > 0
      && this.pending.length >= bootstrapSamples
      && this.pending.length < readySamples
    ) {
      const playbackPcm = this.pending.slice(0, bootstrapSamples);
      output.push(this.makeChunk(playbackPcm, playbackPcm, false, {
        geometryFinal: true,
        bootstrap: true,
      }));
      this.commitPlayback(playbackPcm, contextSamples);
      this.pending = this.pending.slice(bootstrapSamples);
      this.bootstrapEmitted = true;
    }

    while (this.pending.length >= readySamples) {
      const playbackPcm = this.pending.slice(0, payloadSamples);
      const currentAndFuture = this.pending.slice(0, readySamples);
      output.push(this.makeChunk(playbackPcm, currentAndFuture, false));
      this.commitPlayback(playbackPcm, contextSamples);
      this.pending = this.pending.slice(payloadSamples);
      this.bootstrapEmitted = true;
    }

    if (final) {
      this.ended = true;
      while (this.pending.length > payloadSamples) {
        const playbackPcm = this.pending.slice(0, payloadSamples);
        const currentAndFuture = this.pending.slice(
          0,
          Math.min(this.pending.length, readySamples),
        );
        output.push(this.makeChunk(playbackPcm, currentAndFuture, false));
        this.commitPlayback(playbackPcm, contextSamples);
        this.pending = this.pending.slice(payloadSamples);
        this.bootstrapEmitted = true;
      }
      if (this.pending.length > 0) {
        const playbackPcm = this.pending.slice();
        output.push(this.makeChunk(playbackPcm, playbackPcm, true));
        this.pending = new Int16Array();
        this.bootstrapEmitted = true;
      } else if (output.length > 0) {
        output[output.length - 1].final = true;
        output[output.length - 1].geometryFinal = true;
      }
    }
    return output;
  }

  private makeChunk(
    playbackPcm: Int16Array,
    currentAndFuture: Int16Array,
    final: boolean,
    flags: { geometryFinal?: boolean; bootstrap?: boolean } = {},
  ): StreamingRenderChunk {
    const history = this.mode === "prefix" ? this.prefix : this.past;
    const discardFrames = this.mode === "prefix"
      ? this.nextFrameOffset
      : Math.floor(history.length / STREAM_SAMPLES_PER_FRAME);
    const outputFrames = Math.max(
      1,
      Math.ceil(playbackPcm.length / STREAM_SAMPLES_PER_FRAME),
    );
    const chunk: StreamingRenderChunk = {
      modelPcm: concatenate(history, currentAndFuture),
      playbackPcm,
      discardFrames,
      outputFrames,
      frameOffset: this.nextFrameOffset,
      final,
      geometryFinal: flags.geometryFinal ?? final,
      bootstrap: flags.bootstrap ?? false,
    };
    this.nextFrameOffset += outputFrames;
    return chunk;
  }

  private commitPlayback(playbackPcm: Int16Array, contextSamples: number): void {
    if (this.mode === "prefix") {
      this.prefix = concatenate(this.prefix, playbackPcm);
      return;
    }
    const history = concatenate(this.past, playbackPcm);
    this.past = history.slice(Math.max(0, history.length - contextSamples));
  }
}
