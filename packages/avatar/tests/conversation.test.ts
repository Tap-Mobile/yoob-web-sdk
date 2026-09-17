import { test } from "node:test";
import assert from "node:assert/strict";
import { YoobConversation } from "../src/conversation";
import type { YoobAvatar } from "../src/index";

class FakeSocket {
  static OPEN = 1;
  static last?: FakeSocket;
  readyState = 1;
  sent: Array<Record<string, unknown>> = [];
  onopen?: () => void;
  onmessage?: (event: { data: string }) => void;
  onerror?: () => void;
  onclose?: (event: { code: number }) => void;
  constructor(readonly url: string, readonly protocols: string[]) {
    FakeSocket.last = this;
    queueMicrotask(() => this.onopen?.());
  }
  send(data: string) { this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; }
  server(event: object) { this.onmessage?.({ data: JSON.stringify(event) }); }
}
(globalThis as unknown as { WebSocket: typeof FakeSocket }).WebSocket = FakeSocket;

function fakeAvatar() {
  const calls: string[] = [];
  const listeners: Array<(pcm: Int16Array) => void> = [];
  let heard = 0;
  const avatar = {
    prepare: async () => undefined,
    unlockAudio: async () => undefined,
    speak: (pcm: Int16Array) => { calls.push(`speak:${pcm.length}`); heard = 1234; },
    endSpeech: () => calls.push("end"),
    interrupt: () => { calls.push("interrupt"); const h = heard; heard = 0; return h; },
    microphone: {
      on: (_: string, fn: (pcm: Int16Array) => void) => { listeners.push(fn); return () => undefined; },
      start: async () => calls.push("mic-start"),
      stop: () => calls.push("mic-stop"),
    },
  };
  return { avatar: avatar as unknown as YoobAvatar, calls, mic: (pcm: Int16Array) => listeners.forEach((l) => l(pcm)) };
}

const b64 = (samples: number[]) => Buffer.from(new Int16Array(samples).buffer).toString("base64");

test("configures a fast, echo-safe session and streams mic audio", async () => {
  const { avatar, mic } = fakeAvatar();
  const convo = new YoobConversation(avatar, { getClientSecret: async () => "ek_test", voice: "marin" });
  await convo.start();
  const socket = FakeSocket.last!;
  assert.deepEqual(socket.protocols, ["realtime", "openai-insecure-api-key.ek_test"]);
  const update = socket.sent[0] as { session: { audio: { input: Record<string, unknown>; output: Record<string, unknown> } } };
  assert.equal(update.session.audio.input.turn_detection && (update.session.audio.input.turn_detection as { type: string }).type, "server_vad");
  assert.deepEqual(update.session.audio.input.noise_reduction, { type: "far_field" });
  assert.equal(update.session.audio.output.speed, 1.08);
  mic(new Int16Array([1, 2]));
  assert.equal(socket.sent.at(-1)?.type, "input_audio_buffer.append");
  assert.equal(socket.sent.at(-1)?.audio, b64([1, 2]));
  assert.equal(convo.state, "listening");
});

test("plays the active reply and barges in with an exact truncate", async () => {
  const { avatar, calls } = fakeAvatar();
  const states: string[] = [];
  const convo = new YoobConversation(avatar, { getClientSecret: async () => "ek", onState: (s) => states.push(s) });
  await convo.start();
  const socket = FakeSocket.last!;
  socket.server({ type: "response.created", response: { id: "r1" } });
  socket.server({ type: "response.output_item.added", response_id: "r1", item: { id: "i1" } });
  socket.server({ type: "response.output_audio.delta", response_id: "r1", delta: b64([5, 6, 7]) });
  assert.ok(calls.includes("speak:3"));
  assert.equal(convo.state, "speaking");
  socket.sent.length = 0;
  socket.server({ type: "input_audio_buffer.speech_started" });
  assert.ok(calls.includes("interrupt"));
  assert.deepEqual(socket.sent.map((e) => e.type), ["response.cancel", "conversation.item.truncate"]);
  assert.equal(socket.sent[1].audio_end_ms, 1234);
  // Late audio from the cancelled reply is ignored.
  const before = calls.length;
  socket.server({ type: "response.output_audio.delta", response_id: "r1", delta: b64([1]) });
  assert.equal(calls.length, before);
  assert.equal(convo.state, "listening");
});

test("finishes incomplete replies but drops failed ones", async () => {
  const { avatar, calls } = fakeAvatar();
  const convo = new YoobConversation(avatar, { getClientSecret: async () => "ek" });
  await convo.start();
  const socket = FakeSocket.last!;
  socket.server({ type: "response.created", response: { id: "r2" } });
  socket.server({ type: "response.output_audio.delta", response_id: "r2", delta: b64([1, 1]) });
  socket.server({ type: "response.done", response: { id: "r2", status: "incomplete" } });
  assert.equal(calls.at(-1), "end");
  assert.ok(!calls.slice(-2).includes("interrupt"));
  socket.server({ type: "response.created", response: { id: "r3" } });
  socket.server({ type: "response.done", response: { id: "r3", status: "failed" } });
  assert.ok(calls.slice(-2).includes("interrupt"));
  convo.stop();
  assert.ok(calls.includes("mic-stop"));
  assert.equal(convo.state, "ended");
});
