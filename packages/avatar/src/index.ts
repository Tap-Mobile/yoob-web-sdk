import {
  ChunkStore, SDK_VERSION, YoobError, clearChunkCache, fetchManifest, pruneChunkCache,
  type CharacterManifest,
} from "./cdn";
import type { RemoteAudioTrack } from "./engine/audio/conversation-audio";
import { RenderCoordinator } from "./engine/runtime/render-coordinator";
import type { RendererSpatialContract } from "./engine/runtime/generated/runtime-tier-contract";
import type { RendererTemporalContract } from "./engine/runtime/renderer-temporal";

import { YoobMicrophone } from "./microphone";

export { YoobError, type CharacterManifest };
export { YoobMicrophone, type MicrophoneOption, type MicrophoneState, type MicrophoneEvents } from "./microphone";
export {
  YoobConversation, type YoobConversationOptions, type ConversationState, type TurnDetection,
} from "./conversation";
export const version = SDK_VERSION;

/**
 * The methods of a LiveKit `RemoteAudioTrack` that `attachAudioTrack()` uses. Any livekit-client 2.x remote audio track
 * fits; the core package does not depend on livekit-client.
 */
export type YoobAudioTrack = RemoteAudioTrack;

/** What your backend returns from `POST /api/v1/avatar/sessions`. Never put your Yoob API key in a web page. */
export interface YoobCredentials {
  session_token: string;
  download_token: string;
  heartbeat_seconds?: number;
  api_base?: string;
  cdn_base?: string;
}

export type YoobPhase =
  | "not-prepared" | "downloading" | "warming" | "ready" | "speaking" | "failed" | "stopped";

export interface YoobProgress {
  completedBytes: number;
  totalBytes: number;
  fraction: number;
}

export interface YoobAvatarOptions {
  /** Element the character is drawn into. It fills the element, cropping to keep its aspect ratio. */
  container: HTMLElement;
  /** Character id, for example `"luna-anime"`. */
  character: string;
  /** Pin a character version. The newest compatible version is used by default. */
  version?: string;
  /** Asks your backend for a Yoob session. Called again when a session expires. */
  getCredentials: () => Promise<YoobCredentials>;
  /** `"cover"` (default) fills the container; `"contain"` letterboxes. */
  fit?: "cover" | "contain";
  onPhase?: (phase: YoobPhase) => void;
  onProgress?: (progress: YoobProgress) => void;
  onError?: (error: YoobError) => void;
}

export interface YoobSupport {
  supported: boolean;
  reason?: string;
}

const ORT_WASM_PATH = "v1/runtime/onnxruntime-web-1.27.0/ort-wasm-simd-threaded.asyncify.wasm";
const SAMPLE_RATE = 24_000;

/**
 * A talking character in the page. Create it, call `prepare()`, then pass speech to `speak()`.
 * The avatar plays the audio itself so the lips stay in sync.
 */
export class YoobAvatar {
  private phaseValue: YoobPhase = "not-prepared";
  private readonly root: HTMLDivElement;
  private readonly poster: HTMLImageElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly video: HTMLVideoElement;
  private coordinator?: RenderCoordinator;
  private credentials?: YoobCredentials;
  private manifestValue?: CharacterManifest;
  private preparing?: Promise<void>;
  private heartbeat?: ReturnType<typeof setInterval>;
  private objectUrls: string[] = [];
  private utterance = 0;
  private speaking = false;
  private ended = false;
  private destroyed = false;
  private runtimeEvent?: (event: { loadedBytes?: number }) => void;
  /** The user's microphone: device choice, mute, level and audio packets. */
  readonly microphone = new YoobMicrophone(() => this.engine().audio);

  constructor(private readonly options: YoobAvatarOptions) {
    const fit = options.fit ?? "cover";
    this.root = document.createElement("div");
    this.root.className = "yoob-avatar";
    this.root.setAttribute("role", "img");
    this.root.setAttribute("aria-label", "Character");
    Object.assign(this.root.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden" });
    const layer = { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: fit };
    this.poster = document.createElement("img");
    this.poster.alt = "";
    this.poster.decoding = "async";
    Object.assign(this.poster.style, layer);
    this.canvas = document.createElement("canvas");
    this.canvas.width = 1080;
    this.canvas.height = 1920;
    Object.assign(this.canvas.style, layer, { opacity: "0", transition: "opacity 200ms ease" });
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.preload = "auto";
    this.video.setAttribute("aria-hidden", "true");
    // The engine draws the video into the canvas; it stays in the document so browsers keep decoding it.
    Object.assign(this.video.style, { position: "absolute", width: "1px", height: "1px", opacity: "0", pointerEvents: "none" });
    this.root.append(this.poster, this.canvas, this.video);
    options.container.append(this.root);
  }

  /** Whether this browser can render characters: WebGPU plus the Web Crypto and Audio APIs the SDK uses. */
  static async isSupported(): Promise<YoobSupport> {
    if (typeof navigator === "undefined" || !("gpu" in navigator)) {
      return { supported: false, reason: "This browser has no WebGPU. Use a current Chrome or Edge on a desktop." };
    }
    try {
      const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) return { supported: false, reason: "WebGPU is turned off or has no usable graphics adapter." };
    } catch {
      return { supported: false, reason: "WebGPU is unavailable." };
    }
    if (typeof AudioWorkletNode === "undefined") return { supported: false, reason: "This browser has no AudioWorklet." };
    return { supported: true };
  }

  static clearCache(): Promise<void> {
    return clearChunkCache();
  }

  get phase(): YoobPhase { return this.phaseValue; }
  get manifest(): CharacterManifest | undefined { return this.manifestValue; }
  /** Milliseconds of the current utterance the listener has heard. */
  get playedMs(): number { return this.coordinator?.playedAudioMs ?? 0; }

  /**
   * Downloads what is missing and starts the renderer. The poster shows within the first request or two and the idle
   * loop soon after; the models finish behind it. Safe to call again after a failure.
   */
  prepare(): Promise<void> {
    if (this.phaseValue === "ready" || this.phaseValue === "speaking") return Promise.resolve();
    this.preparing ??= this.load().catch((error: unknown) => {
      const failure = toYoobError(error);
      this.setPhase("failed");
      this.options.onError?.(failure);
      throw failure;
    }).finally(() => { this.preparing = undefined; });
    return this.preparing;
  }

  /**
   * Call from a click or key press before the first `speak()`, so the browser allows sound. `speak()` also tries.
   */
  async unlockAudio(): Promise<void> {
    await this.engine().audio.activatePlayback();
  }

  /**
   * Plays 24 kHz mono 16-bit PCM and moves the face with it. Call for each chunk as it streams in, then `endSpeech()`.
   */
  speak(pcm: Int16Array | ArrayBuffer, sampleRate = SAMPLE_RATE): void {
    if (this.phaseValue === "stopped") throw new YoobError("out-of-credit", "The Yoob session has stopped.");
    if (sampleRate !== SAMPLE_RATE) {
      throw new YoobError("invalid-audio", "@yoob/avatar 0.1 accepts 24 kHz audio. Request 24 kHz PCM from your voice provider.");
    }
    const samples = pcm instanceof Int16Array ? pcm : new Int16Array(pcm);
    if (samples.length === 0) return;
    const coordinator = this.engine();
    if (!this.speaking || this.ended) this.beginUtterance();
    coordinator.appendStreamingAudio(this.responseId, samples, false);
  }

  /** Marks the end of the current utterance. The face returns to idle when the audio finishes. */
  endSpeech(): void {
    if (!this.speaking || this.ended) return;
    this.ended = true;
    this.engine().appendStreamingAudio(this.responseId, new Int16Array(), true);
  }

  /** Stops speaking at once. Returns the milliseconds that were heard, for truncating a realtime reply. */
  interrupt(): number {
    if (!this.coordinator || !this.speaking) return 0;
    const heard = this.coordinator.playedAudioMs;
    this.coordinator.cancel();
    this.finishUtterance();
    return heard;
  }

  /**
   * Listens to a LiveKit remote audio track instead of playing it: `onAudio` receives its sound as 24 kHz mono PCM16 in
   * 20 ms packets (silence included), and the track's own output is muted. Pass the packets you want heard to
   * `speak()`. Only one track is attached at a time. Resolves to a function that detaches the track.
   * `YoobLiveKitSession` from `@yoob/avatar/livekit` does all of this for a LiveKit agent.
   */
  async attachAudioTrack(track: YoobAudioTrack, onAudio: (pcm: Int16Array) => void): Promise<() => void> {
    const audio = this.engine().audio;
    const cleanup = await audio.tapRemoteTrack(track, onAudio);
    return () => audio.stopRemoteTap(cleanup);
  }

  /** Ends the metered session and removes the character from the page. Downloaded files stay cached. */
  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.stopHeartbeat();
    this.microphone.stop();
    this.coordinator?.destroy();
    this.coordinator = undefined;
    for (const url of this.objectUrls) URL.revokeObjectURL(url);
    this.root.remove();
    if (this.credentials) await this.session("end").catch(() => undefined);
    this.setPhase("not-prepared");
  }

  private get responseId(): string {
    return `yoob-${this.utterance}`;
  }

  private beginUtterance(): void {
    this.utterance += 1;
    this.speaking = true;
    this.ended = false;
    void this.engine().audio.activatePlayback().catch(() => undefined);
    this.setPhase("speaking");
  }

  private finishUtterance(): void {
    this.speaking = false;
    this.ended = false;
    if (this.phaseValue === "speaking") this.setPhase(this.coordinator ? "ready" : "not-prepared");
  }

  private engine(): RenderCoordinator {
    if (this.destroyed) throw new YoobError("renderer", "This avatar was destroyed.");
    this.coordinator ??= new RenderCoordinator(this.canvas, this.video, {
      onFirstHostFrame: () => { this.canvas.style.opacity = "1"; },
      onPlaybackEnded: () => this.finishUtterance(),
      onError: (message) => this.options.onError?.(new YoobError("renderer", message)),
      onRuntimeEvent: (event) => this.runtimeEvent?.(event),
    });
    return this.coordinator;
  }

  private async load(): Promise<void> {
    const support = await YoobAvatar.isSupported();
    if (!support.supported) throw new YoobError("unsupported", support.reason ?? "WebGPU is required.");
    this.setPhase("downloading");
    this.credentials = await this.options.getCredentials();
    const access = { cdnBase: this.cdnBase, downloadToken: this.credentials.download_token };
    const manifest = await fetchManifest(access, this.options.character, this.options.version, "web");
    this.manifestValue = manifest;
    this.root.setAttribute("aria-label", manifest.displayName);
    const runtime = manifest.runtime;
    if (!runtime || manifest.engine !== "anime-web") {
      throw new YoobError("unsupported", `${manifest.character} has no web renderer in this SDK version.`);
    }

    const store = new ChunkStore(access, manifest);
    const total = manifest.files.reduce((sum, file) => sum + file.size, 0);
    let completed = 0;
    const count = (bytes: number) => {
      completed += bytes;
      this.options.onProgress?.({ completedBytes: completed, totalBytes: total, fraction: total ? completed / total : 0 });
    };

    // Tier 0 and 1 on the page: the poster, then the idle loop the engine draws.
    const poster = await store.bytes(manifest.poster, count);
    this.poster.src = this.objectUrl(poster, "image/jpeg");
    const videoFile = manifest.files.find((file) => file.tier === 1 && file.path.endsWith(".mp4"));
    if (!videoFile) throw new YoobError("invalid-assets", "The character has no idle video.");
    const video = await store.bytes(videoFile.path, count);
    this.video.src = this.objectUrl(video, "video/mp4");
    this.video.load();
    // The worker downloads the rest; count everything else as it lands.
    const workerBytes = total - completed;
    const coordinator = this.engine();
    const workerStart = completed;

    this.setPhase("warming");
    const statusSink = (event: { loadedBytes?: number }) => {
      if (event.loadedBytes === undefined) return;
      const done = workerStart + Math.min(workerBytes, event.loadedBytes);
      this.options.onProgress?.({ completedBytes: done, totalBytes: total, fraction: total ? done / total : 0 });
    };
    this.runtimeEvent = statusSink;
    await coordinator.initialize(
      { ...access, manifest, ortWasmUrl: `${this.cdnBase}/${ORT_WASM_PATH}` },
      runtime.neuralMouthStride ?? 1,
      runtime.rendererInputType ?? "float32",
      runtime.rendererPreferredLayout ?? "NCHW",
      runtime.rendererSpatialContract as RendererSpatialContract | undefined,
      runtime.rendererTemporalContract as RendererTemporalContract | undefined,
    );
    this.options.onProgress?.({ completedBytes: total, totalBytes: total, fraction: 1 });
    this.canvas.style.opacity = "1";
    void pruneChunkCache([manifest]).catch(() => undefined);
    this.startHeartbeat();
    this.setPhase("ready");
  }

  private get cdnBase(): string {
    return (this.credentials?.cdn_base ?? "https://cdn.yoob.com").replace(/\/+$/, "");
  }

  private objectUrl(data: ArrayBuffer, type: string): string {
    const url = URL.createObjectURL(new Blob([data], { type }));
    this.objectUrls.push(url);
    return url;
  }

  private setPhase(phase: YoobPhase): void {
    if (this.phaseValue === phase) return;
    this.phaseValue = phase;
    this.options.onPhase?.(phase);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    const seconds = Math.max(5, this.credentials?.heartbeat_seconds ?? 15);
    this.heartbeat = setInterval(() => void this.beat(), seconds * 1000);
    document.addEventListener("visibilitychange", this.onVisibility);
    addEventListener("pagehide", this.onPageHide);
  }

  private stopHeartbeat(): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    this.heartbeat = undefined;
    document.removeEventListener("visibilitychange", this.onVisibility);
    removeEventListener("pagehide", this.onPageHide);
  }

  private readonly onVisibility = () => {
    if (document.visibilityState === "visible" && this.phaseValue !== "stopped") void this.beat();
  };

  private readonly onPageHide = () => {
    if (!this.credentials) return;
    // Best effort: keepalive lets the final beat outlive the page.
    void fetch(`${this.apiBase}/api/v1/sessions/end`, {
      method: "POST", keepalive: true, headers: { authorization: `Bearer ${this.credentials.session_token}` },
    }).catch(() => undefined);
  };

  private get apiBase(): string {
    return (this.credentials?.api_base ?? "https://api2.yoob.com").replace(/\/+$/, "");
  }

  private async beat(): Promise<void> {
    try {
      const reply = await this.session("heartbeat");
      if (!reply.stop) return;
      if (reply.reason === "out-of-credits") {
        this.interrupt();
        this.stopHeartbeat();
        this.setPhase("stopped");
        this.options.onError?.(new YoobError("out-of-credit", "This Yoob workspace is out of credit."));
      } else {
        this.credentials = await this.options.getCredentials();
      }
    } catch (error) {
      // The console ends sessions that stopped beating (a sleeping laptop): open a new one.
      if (error instanceof YoobError && error.code === "unauthorized") {
        this.credentials = await this.options.getCredentials().catch(() => this.credentials);
      }
    }
  }

  private async session(action: "heartbeat" | "end"): Promise<{ stop?: boolean; reason?: string | null }> {
    const response = await fetch(`${this.apiBase}/api/v1/sessions/${action}`, {
      method: "POST",
      headers: { authorization: `Bearer ${this.credentials?.session_token ?? ""}`, "x-yoob-sdk": `yoob-web/${SDK_VERSION}` },
    });
    if (response.status === 401 || response.status === 404) throw new YoobError("unauthorized", "Session ended.");
    if (!response.ok) throw new YoobError("network", `Heartbeat failed (HTTP ${response.status}).`);
    return response.json();
  }
}

function toYoobError(error: unknown): YoobError {
  if (error instanceof YoobError) return error;
  return new YoobError("renderer", error instanceof Error ? error.message : String(error));
}
