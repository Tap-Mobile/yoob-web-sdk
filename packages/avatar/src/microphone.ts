import type { ConversationAudio } from "./engine/audio/conversation-audio";
import { enumerateMicrophones, type MicrophoneOption } from "./engine/audio/microphone-devices";
import { YoobError } from "./cdn";

export type { MicrophoneOption };

export type MicrophoneState = "off" | "starting" | "live" | "muted" | "failed";

export interface MicrophoneEvents {
  /** 24 kHz mono PCM16 in 20 ms packets, echo-cancelled against the character's voice. Not sent while muted. */
  audio: (pcm: Int16Array) => void;
  /** Input level from 0 to 1, about 20 times a second, for a meter. 0 while muted. */
  level: (level: number) => void;
  state: (state: MicrophoneState) => void;
  /** Inputs were plugged in or removed. */
  devices: (devices: MicrophoneOption[]) => void;
  /** A readable explanation of what went wrong and what to do. */
  error: (error: YoobError) => void;
}

type Listener<K extends keyof MicrophoneEvents> = MicrophoneEvents[K];

/**
 * The user's microphone, captured through the same audio graph that plays the character, so the browser's echo
 * canceller removes the character's voice. A stalled input is reopened automatically.
 */
export class YoobMicrophone {
  private stateValue: MicrophoneState = "off";
  private deviceIdValue: string | null = null;
  private listeners: { [K in keyof MicrophoneEvents]: Set<Listener<K>> } = {
    audio: new Set(), level: new Set(), state: new Set(), devices: new Set(), error: new Set(),
  };
  private levelValue = 0;
  private lastLevelAt = 0;
  private deviceWatch?: () => void;

  constructor(private readonly audio: () => ConversationAudio) {}

  get state(): MicrophoneState { return this.stateValue; }
  get muted(): boolean { return this.stateValue === "muted"; }
  /** The selected input, or null for the system default. */
  get deviceId(): string | null { return this.deviceIdValue; }
  get level(): number { return this.levelValue; }

  on<K extends keyof MicrophoneEvents>(event: K, listener: MicrophoneEvents[K]): () => void {
    this.listeners[event].add(listener);
    return () => this.listeners[event].delete(listener);
  }

  /**
   * Available inputs. Labels are empty until the user has granted microphone access once, so call this again after
   * `start()`.
   */
  async devices(): Promise<MicrophoneOption[]> {
    if (!navigator.mediaDevices) return [];
    return enumerateMicrophones(navigator.mediaDevices);
  }

  /** Asks for permission if needed and starts capturing. Call from a click so audio can start too. */
  async start(options: { deviceId?: string | null } = {}): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw this.fail(new YoobError("unsupported", "This browser can't use a microphone here. Microphones need HTTPS."));
    }
    if (options.deviceId !== undefined) this.deviceIdValue = options.deviceId;
    const audio = this.audio();
    this.setState("starting");
    audio.onMicPcm = (buffer) => this.packet(buffer);
    audio.onMicrophoneFailure = (message) => {
      this.fail(new YoobError("renderer", `${message}. Check the input and press the microphone button again.`));
    };
    audio.onMicrophoneRecovered = () => undefined;
    try {
      await audio.activatePlayback();
      await audio.enableMicrophone(this.deviceIdValue);
    } catch (error) {
      throw this.fail(microphoneError(error));
    }
    audio.startMicrophoneWatchdog(this.deviceIdValue);
    this.watchDevices();
    this.setState("live");
  }

  /** Switches input without interrupting a conversation. Falls back to the default if the device is gone. */
  async select(deviceId: string | null): Promise<void> {
    this.deviceIdValue = deviceId;
    if (this.stateValue !== "live" && this.stateValue !== "muted") return;
    const audio = this.audio();
    try {
      const { usedFallback } = await audio.switchMicrophone(deviceId);
      if (usedFallback) this.deviceIdValue = null;
      audio.startMicrophoneWatchdog(this.deviceIdValue);
      if (this.stateValue === "muted") audio.disableMicrophone();
    } catch (error) {
      throw this.fail(microphoneError(error));
    }
  }

  /** Mutes or unmutes. While muted nothing is captured or sent. */
  setMuted(muted: boolean): void {
    if (this.stateValue !== "live" && this.stateValue !== "muted") return;
    const audio = this.audio();
    if (muted) {
      audio.disableMicrophone();
      this.emitLevel(0);
      this.setState("muted");
    } else {
      void audio.enableMicrophone(this.deviceIdValue).then(() => {
        audio.startMicrophoneWatchdog(this.deviceIdValue);
        this.setState("live");
      }).catch((error: unknown) => this.fail(microphoneError(error)));
    }
  }

  /** Stops capturing and releases the device (the browser's recording indicator turns off). */
  stop(): void {
    if (this.stateValue === "off") return;
    const audio = this.audio();
    audio.disableMicrophone();
    audio.releaseMicrophone();
    this.deviceWatch?.();
    this.deviceWatch = undefined;
    this.emitLevel(0);
    this.setState("off");
  }

  private packet(buffer: ArrayBuffer): void {
    if (this.stateValue !== "live") return;
    const pcm = new Int16Array(buffer);
    for (const listener of this.listeners.audio) listener(pcm);
    const now = performance.now();
    if (now - this.lastLevelAt < 50) return;
    this.lastLevelAt = now;
    let sum = 0;
    for (let i = 0; i < pcm.length; i += 1) sum += (pcm[i] / 32768) ** 2;
    this.emitLevel(Math.min(1, Math.sqrt(sum / Math.max(1, pcm.length)) * 4));
  }

  private emitLevel(level: number): void {
    this.levelValue = level;
    for (const listener of this.listeners.level) listener(level);
  }

  private setState(state: MicrophoneState): void {
    if (this.stateValue === state) return;
    this.stateValue = state;
    for (const listener of this.listeners.state) listener(state);
  }

  private fail(error: YoobError): YoobError {
    this.setState("failed");
    this.emitLevel(0);
    for (const listener of this.listeners.error) listener(error);
    return error;
  }

  private watchDevices(): void {
    if (this.deviceWatch || !navigator.mediaDevices?.addEventListener) return;
    const onChange = () => {
      void this.devices().then((devices) => {
        for (const listener of this.listeners.devices) listener(devices);
        // The selected input was unplugged: move to the default one.
        if (this.deviceIdValue && !devices.some((d) => d.deviceId === this.deviceIdValue)) void this.select(null);
      });
    };
    navigator.mediaDevices.addEventListener("devicechange", onChange);
    this.deviceWatch = () => navigator.mediaDevices.removeEventListener("devicechange", onChange);
  }
}

export function microphoneError(error: unknown): YoobError {
  if (error instanceof YoobError) return error;
  const name = typeof error === "object" && error && "name" in error ? String(error.name) : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return new YoobError("unauthorized", "Microphone access is blocked. Allow the microphone for this site and try again.");
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return new YoobError("unsupported", "No microphone was found. Connect one or choose another input.");
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return new YoobError("renderer", "The microphone is busy in another app or couldn't be opened.");
  }
  return new YoobError("renderer", error instanceof Error ? error.message : "The microphone couldn't be opened.");
}
