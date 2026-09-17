import { YoobError } from "./cdn";
import type { YoobAvatar } from "./index";

/** Turn-taking settings passed to OpenAI Realtime. */
export type TurnDetection =
  | { type: "server_vad"; silenceMs?: number; threshold?: number; prefixPaddingMs?: number }
  | { type: "semantic_vad"; eagerness?: "low" | "medium" | "high" | "auto" };

export interface YoobConversationOptions {
  /**
   * Returns a short-lived OpenAI Realtime client secret from your backend
   * (`POST https://api.openai.com/v1/realtime/client_secrets`). Never put your OpenAI key in a page.
   */
  getClientSecret: () => Promise<string>;
  model?: string;
  voice?: string;
  instructions?: string;
  /** Spoken speed, 0.25–1.5. Default 1.08, which the Yoob demo measured as natural but snappy. */
  speed?: number;
  /**
   * Default `server_vad` with a 450 ms silence window: replies start about 0.8 s sooner than `semantic_vad`, which
   * waits to judge whether a sentence is finished. Raise `threshold` for noisy rooms instead of muting the mic.
   */
  turnDetection?: TurnDetection;
  /** `far_field` (default) suits laptops and kiosks; `near_field` suits headsets. */
  noiseReduction?: "far_field" | "near_field" | null;
  /** Transcribe what the user says (for captions). Default `gpt-4o-mini-transcribe`; null turns it off. */
  transcriptionModel?: string | null;
  /** Have the character speak first. */
  greet?: boolean;
  onState?: (state: ConversationState) => void;
  /** What the user is saying; `final` once the turn is transcribed. */
  onUserTranscript?: (text: string, final: boolean) => void;
  /** What the character is saying, as it streams. */
  onAssistantTranscript?: (text: string, final: boolean) => void;
  onError?: (error: YoobError) => void;
}

export type ConversationState = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "ended";

const REALTIME_URL = "wss://api.openai.com/v1/realtime";

interface ServerEvent {
  type: string;
  delta?: string;
  transcript?: string;
  response_id?: string;
  item_id?: string;
  response?: { id?: string; status?: string };
  item?: { id?: string };
  error?: { code?: string; type?: string; message?: string };
}

/**
 * A spoken conversation between the user and a Yoob character, using OpenAI Realtime.
 * Microphone audio goes from the browser to OpenAI; replies stream into the avatar, which plays them in sync.
 * Speaking over the character interrupts it, and the model is told how much of its reply was heard.
 */
export class YoobConversation {
  private socket?: WebSocket;
  private stateValue: ConversationState = "idle";
  private unsubscribe: Array<() => void> = [];
  private activeResponse?: string;
  private playingItem?: string;
  private readonly finished = new Set<string>();
  private userText = "";
  private assistantText = "";
  private closing = false;

  constructor(private readonly avatar: YoobAvatar, private readonly options: YoobConversationOptions) {}

  get state(): ConversationState { return this.stateValue; }

  /** Starts listening. Call from a click: it asks for the microphone and unlocks sound. */
  async start(options: { deviceId?: string | null } = {}): Promise<void> {
    if (this.socket) return;
    this.closing = false;
    this.setState("connecting");
    try {
      await this.avatar.prepare();
      await this.avatar.unlockAudio();
      const secret = await this.options.getClientSecret();
      await this.open(secret);
      this.configure();
      const microphone = this.avatar.microphone;
      this.unsubscribe.push(microphone.on("audio", (pcm) => this.send({ type: "input_audio_buffer.append", audio: toBase64(pcm) })));
      await microphone.start(options);
      this.setState("listening");
      if (this.options.greet) this.send({ type: "response.create" });
    } catch (error) {
      const failure = error instanceof YoobError ? error : new YoobError("network", errorText(error));
      this.stop();
      this.options.onError?.(failure);
      throw failure;
    }
  }

  /** Sends typed text as the user's turn. */
  sendText(text: string): void {
    if (!text.trim()) return;
    this.bargeIn();
    this.send({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
    });
    this.send({ type: "response.create" });
    this.setState("thinking");
  }

  /** Ends the conversation and releases the microphone. The avatar stays on screen. */
  stop(): void {
    this.closing = true;
    for (const off of this.unsubscribe.splice(0)) off();
    this.avatar.microphone.stop();
    this.avatar.interrupt();
    this.socket?.close(1000, "done");
    this.socket = undefined;
    this.activeResponse = undefined;
    this.playingItem = undefined;
    this.setState("ended");
  }

  private open(secret: string): Promise<void> {
    const model = encodeURIComponent(this.options.model ?? "gpt-realtime");
    return new Promise((resolve, reject) => {
      // Browsers can't set headers on WebSockets; OpenAI accepts an ephemeral secret as a subprotocol.
      const socket = new WebSocket(`${REALTIME_URL}?model=${model}`, ["realtime", `openai-insecure-api-key.${secret}`]);
      const timer = setTimeout(() => { socket.close(); reject(new YoobError("network", "OpenAI Realtime didn't answer in time.")); }, 15_000);
      socket.onopen = () => { clearTimeout(timer); this.socket = socket; resolve(); };
      socket.onerror = () => { clearTimeout(timer); reject(new YoobError("network", "Couldn't connect to OpenAI Realtime. Check the client secret.")); };
      socket.onmessage = (event) => this.handle(String(event.data));
      socket.onclose = (event) => {
        if (this.socket !== socket) return;
        this.socket = undefined;
        if (!this.closing) {
          this.options.onError?.(new YoobError("network", `The conversation disconnected (${event.code}).`));
          this.stop();
        }
      };
    });
  }

  private configure(): void {
    const turn = this.options.turnDetection ?? { type: "server_vad" };
    const turnDetection = turn.type === "semantic_vad"
      ? { type: "semantic_vad", eagerness: turn.eagerness ?? "auto", create_response: true, interrupt_response: true }
      : {
          type: "server_vad",
          silence_duration_ms: turn.silenceMs ?? 450,
          prefix_padding_ms: turn.prefixPaddingMs ?? 300,
          threshold: turn.threshold ?? 0.5,
          create_response: true,
          interrupt_response: true,
        };
    const noise = this.options.noiseReduction === undefined ? "far_field" : this.options.noiseReduction;
    const transcription = this.options.transcriptionModel === undefined ? "gpt-4o-mini-transcribe" : this.options.transcriptionModel;
    this.send({
      type: "session.update",
      session: {
        type: "realtime",
        output_modalities: ["audio"],
        ...(this.options.instructions ? { instructions: this.options.instructions } : {}),
        audio: {
          input: {
            format: { type: "audio/pcm", rate: 24_000 },
            noise_reduction: noise ? { type: noise } : null,
            transcription: transcription ? { model: transcription } : null,
            turn_detection: turnDetection,
          },
          output: {
            format: { type: "audio/pcm", rate: 24_000 },
            ...(this.options.voice ? { voice: this.options.voice } : {}),
            speed: this.options.speed ?? 1.08,
          },
        },
      },
    });
  }

  private handle(frame: string): void {
    let event: ServerEvent;
    try { event = JSON.parse(frame) as ServerEvent; } catch { return; }
    switch (event.type) {
      case "input_audio_buffer.speech_started":
        this.bargeIn();
        this.userText = "";
        this.options.onUserTranscript?.("", false);
        this.setState("listening");
        return;
      case "input_audio_buffer.speech_stopped":
        this.setState("thinking");
        return;
      case "conversation.item.input_audio_transcription.delta":
        this.userText += event.delta ?? "";
        this.options.onUserTranscript?.(this.userText, false);
        return;
      case "conversation.item.input_audio_transcription.completed":
        this.userText = event.transcript ?? this.userText;
        this.options.onUserTranscript?.(this.userText, true);
        return;
      case "response.created":
        if (event.response?.id && !this.finished.has(event.response.id)) {
          this.activeResponse = event.response.id;
          this.assistantText = "";
        }
        return;
      case "response.output_item.added":
        if (event.response_id === this.activeResponse && event.item?.id) this.playingItem = event.item.id;
        return;
      case "response.output_audio.delta": {
        if (!event.delta || !event.response_id || event.response_id !== this.activeResponse) return;
        this.avatar.speak(fromBase64(event.delta));
        this.setState("speaking");
        return;
      }
      case "response.output_audio.done":
        if (event.response_id === this.activeResponse) this.avatar.endSpeech();
        return;
      case "response.output_audio_transcript.delta":
        if (event.response_id !== this.activeResponse) return;
        this.assistantText += event.delta ?? "";
        this.options.onAssistantTranscript?.(this.assistantText, false);
        return;
      case "response.output_audio_transcript.done":
        if (event.response_id === this.activeResponse) this.options.onAssistantTranscript?.(event.transcript ?? this.assistantText, true);
        return;
      case "response.done": {
        const id = event.response?.id;
        if (!id) return;
        this.finished.add(id);
        const status = event.response?.status;
        // A reply cut short by a limit still plays what arrived; a cancelled or failed one is dropped.
        if (id === this.activeResponse && status !== "completed" && status !== "incomplete") this.avatar.interrupt();
        if (id === this.activeResponse) this.avatar.endSpeech();
        if (this.stateValue === "thinking" && id === this.activeResponse) this.setState("listening");
        return;
      }
      case "error": {
        const code = event.error?.code ?? event.error?.type ?? "server_error";
        // Cancelling a reply that just finished is a harmless race.
        if (code.includes("response_cancel") || code.includes("no_active_response")) return;
        this.options.onError?.(new YoobError("network", event.error?.message ?? code));
        return;
      }
      default:
    }
  }

  /** The user started talking: stop the character and tell the model how much of its reply was heard. */
  private bargeIn(): void {
    const heardMs = this.avatar.interrupt();
    if (this.activeResponse) {
      this.finished.add(this.activeResponse);
      this.send({ type: "response.cancel", response_id: this.activeResponse });
      if (this.playingItem) {
        this.send({ type: "conversation.item.truncate", item_id: this.playingItem, content_index: 0, audio_end_ms: Math.max(0, heardMs) });
      }
    }
    this.activeResponse = undefined;
    this.playingItem = undefined;
  }

  private send(event: object): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(event));
  }

  private setState(state: ConversationState): void {
    if (this.stateValue === state) return;
    this.stateValue = state;
    this.options.onState?.(state);
  }
}

export function toBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export function fromBase64(text: string): Int16Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length & ~1);
  for (let i = 0; i < bytes.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
