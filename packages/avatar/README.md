# @yoob/avatar

Talking characters rendered in the browser with WebGPU. You give Yoob speech; it plays the audio and moves the face in
sync, on the visitor's own GPU. Add a microphone and OpenAI Realtime or Gemini Live, and you have a live spoken
conversation with barge-in.

- **Light.** About 20 KB gzipped to start. Rendering workers load when a character is created, and character files
  (37 MB) stream from `cdn.yoob.com` in verified chunks cached in the browser. On a good connection the character
  appears in about half a second, and a returning visitor is ready in about one.
- **Private.** Audio and text go to your voice provider, not to Yoob. See [Network](#network).

Works in current desktop Chrome and Edge. Mobile browsers are not supported yet; `YoobAvatar.isSupported()` tells you
before you load anything.

## Install

```sh
npm install @yoob/avatar
```

## Open a session on your backend

Keep your Yoob API key on the server and hand the page a short-lived session:

```js
// POST /yoob-session on your server
const response = await fetch("https://api2.yoob.com/api/v1/avatar/sessions", {
  method: "POST",
  headers: { authorization: `Bearer ${process.env.YOOB_API_KEY}`, "content-type": "application/json" },
  body: JSON.stringify({ characters: ["luna-anime"] }),
});
return response.json(); // { session_token, download_token, heartbeat_seconds }
```

## Show a character

```ts
import { YoobAvatar } from "@yoob/avatar";

const support = await YoobAvatar.isSupported();
if (!support.supported) showFallback(support.reason);

const avatar = new YoobAvatar({
  container: document.querySelector("#character")!,
  character: "luna-anime",
  getCredentials: () => fetch("/yoob-session", { method: "POST" }).then((r) => r.json()),
  onProgress: ({ fraction }) => (bar.value = fraction),
  onPhase: (phase) => console.log(phase), // downloading → warming → ready ⇄ speaking
});
await avatar.prepare();
```

The character fills its container (`fit: "contain"` letterboxes instead).

## Make it talk

```ts
button.onclick = async () => {
  await avatar.unlockAudio();              // browsers allow sound only after a click
  for await (const chunk of myTtsStream()) {
    avatar.speak(chunk);                   // Int16Array, 24 kHz mono
  }
  avatar.endSpeech();
};

const heardMs = avatar.interrupt();        // stop now; returns what was heard
```

## Talk with it: OpenAI Realtime

```ts
import { YoobConversation } from "@yoob/avatar";

const conversation = new YoobConversation(avatar, {
  getClientSecret: () => fetch("/openai-secret", { method: "POST" }).then((r) => r.json()).then((s) => s.value),
  voice: "marin",
  instructions: "You are Luna, a warm, curious companion.",
  greet: true,
  onUserTranscript: (text) => (userCaption.textContent = text),
  onAssistantTranscript: (text) => (lunaCaption.textContent = text),
  onState: (state) => console.log(state), // connecting → listening → thinking → speaking
});

talkButton.onclick = () => conversation.start();   // asks for the microphone
endButton.onclick = () => conversation.stop();
```

Your backend creates the client secret with
[`POST /v1/realtime/client_secrets`](https://platform.openai.com/docs/api-reference/realtime-sessions). The defaults
come from latency measurements on the Yoob demo:

| Setting | Default | Why |
|---|---|---|
| `turnDetection` | `server_vad`, 450 ms silence | Replies start about 0.8 s sooner than `semantic_vad` |
| `noiseReduction` | `far_field` | Laptop and kiosk microphones |
| `speed` | `1.08` | Natural but snappy |

The microphone stays open while the character speaks, so the user can interrupt; the browser's echo canceller removes
the character's voice. In a noisy room, raise `turnDetection.threshold` rather than muting.

## Talk with it: Gemini Live

Bring your own Gemini voice with `YoobGeminiConversation`. It has the same states, transcripts and barge-in as
`YoobConversation`.

```ts
import { YoobGeminiConversation } from "@yoob/avatar";

const conversation = new YoobGeminiConversation(avatar, {
  getToken: () => fetch("/gemini-token", { method: "POST" }).then((r) => r.json()).then((t) => t.name),
  voice: "Kore",
  systemInstruction: "You are Luna, a warm, curious companion.",
  greet: true,
  onUserTranscript: (text) => (userCaption.textContent = text),
  onAssistantTranscript: (text) => (lunaCaption.textContent = text),
  onState: (state) => console.log(state),
});

talkButton.onclick = () => conversation.start();   // asks for the microphone
endButton.onclick = () => conversation.stop();
```

The page never sees your Gemini API key. Your backend creates a single-use
[ephemeral token](https://ai.google.dev/gemini-api/docs/live-api/ephemeral-tokens) and returns its `name`:

```js
// POST /gemini-token on your server
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const token = await ai.authTokens.create({
  config: {
    uses: 1,
    expireTime: new Date(Date.now() + 30 * 60_000).toISOString(),    // messages stop after this
    newSessionExpireTime: new Date(Date.now() + 60_000).toISOString(), // the page must connect before this
    liveConnectConstraints: { model: "gemini-3.8-live" },              // optional: lock the model
  },
});
return { name: token.name };
```

Without the SDK, call `POST https://generativelanguage.googleapis.com/v1beta/auth_tokens` with the
`x-goog-api-key` header and a body of `{ "uses": 1, "expireTime": "…", "newSessionExpireTime": "…" }`. Settings
locked with `liveConnectConstraints` take precedence over the ones the page sends; Google's
[`lockAdditionalFields`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.lock_additional_fields)
controls which. Lock at least the model, so a leaked token can't be used for anything else.

The conversation connects to Gemini's `BidiGenerateContentConstrained` WebSocket with the token. It converts the
microphone's 24 kHz audio to the 16 kHz Gemini expects, and plays Gemini's 24 kHz replies through the avatar.

| Option | Default | Why |
|---|---|---|
| `model` | `gemini-3.8-live` | Google's recommended low-latency native-audio Live model |
| `voice` | Gemini's choice | Any prebuilt voice name, for example `Kore` or `Puck` |
| `activityDetection.startSensitivity` | `"high"` | Quick barge-in. Use `"low"` in noisy rooms. |
| `activityDetection.endSensitivity` | `"high"` | Ends the user's turn sooner |
| `activityDetection.silenceDurationMs` | `450` | The silence window Yoob measured as fastest with OpenAI |
| `activityDetection.prefixPaddingMs` | `100` | Short enough for one-word answers |
| `inputTranscription` / `outputTranscription` | `true` | Captions for both sides |

`sendText(text)` sends a typed turn. `onGoAway(timeLeft)` tells you when Gemini is about to close the connection:
audio-only sessions last up to 15 minutes. Gemini doesn't report when a user turn ends, so a spoken turn goes from
`listening` straight to `speaking`. `thinking` appears after `greet` and `sendText`.

## LiveKit agents

`@yoob/avatar/livekit` shows a [LiveKit](https://docs.livekit.io/agents/) voice agent as a Yoob character. The agent's
audio track is decoded in the browser and drives the face on the visitor's GPU: there is no avatar worker in the room
and no video track, so nothing extra runs on your servers and the call uses only audio bandwidth. The agent needs
nothing special; any normal voice agent works.

```sh
npm install @yoob/avatar livekit-client
```

```ts
import { Room } from "livekit-client";
import { YoobLiveKitSession } from "@yoob/avatar/livekit";

const room = new Room();
const session = new YoobLiveKitSession(avatar, {
  room,
  onUserTranscript: (text) => (userCaption.textContent = text),
  onAssistantTranscript: (text) => (lunaCaption.textContent = text),
  onState: (state) => console.log(state), // connecting → listening → thinking → speaking
});

talkButton.onclick = async () => {
  const sound = avatar.unlockAudio();     // while the click still allows sound
  const { url, token } = await fetch("/livekit-token", { method: "POST" }).then((r) => r.json());
  await room.connect(url, token);
  await sound;
  await session.start();                  // publishes the microphone
};
endButton.onclick = async () => {
  await session.stop();
  await room.disconnect();
};
```

The session follows the first agent in the room (or `agentIdentity`). It splits the agent's speech into replies with
the agent's `lk.agent.state` attribute, or with 600 ms of silence (`silenceMs`) when the agent does not publish one. If
the agent stops speaking while the user is talking, the character stops at once. The microphone is published through
LiveKit with echo cancellation; pass `microphone: { deviceId }` to pick an input, or `microphone: false` to publish it
yourself. Captions come from the agent's `lk.transcription` text streams.

The avatar plays the agent's voice, so don't also attach that track yourself (for example with `RoomAudioRenderer`),
or it will be heard twice. `livekit-client` 2.9 or newer is required only for this entry point; the main package does
not include it.

To handle a track yourself, `avatar.attachAudioTrack(track, (pcm) => avatar.speak(pcm))` mutes a LiveKit
`RemoteAudioTrack` and hands you its sound as 24 kHz PCM. Call `endSpeech()` when a reply ends.

## Microphone controls

`avatar.microphone` handles input selection, mute and level:

```ts
const mic = avatar.microphone;
select.replaceChildren(...(await mic.devices()).map((d) => new Option(d.label, d.deviceId)));
select.onchange = () => mic.select(select.value || null);     // switches mid-conversation
muteButton.onclick = () => mic.setMuted(!mic.muted);
mic.on("level", (level) => (meter.style.width = `${level * 100}%`));
mic.on("devices", refreshList);                                // plugged in or removed
mic.on("error", (error) => (status.textContent = error.message));
```

A stalled input is detected within about 1.6 s and reopened automatically. An unplugged device falls back to the
system default. Errors explain what to do next, for example "Microphone access is blocked. Allow the microphone for
this site and try again."

Using your own voice stack? Call `mic.start()` and read `mic.on("audio", pcm => …)` (24 kHz PCM16, 20 ms packets).

## Characters

| Id | Style | Download |
|---|---|---|
| `luna-anime` | Anime | 37 MB |

`luna-realistic` is available in the [iOS SDK](https://github.com/Tap-Mobile/yoob-ios-sdk). Its web renderer is on the way.

## Network

| Request | When | Contents |
|---|---|---|
| `cdn.yoob.com` character files | First visit and version updates | Download grant |
| `cdn.yoob.com` ONNX Runtime WebAssembly | First visit | Nothing |
| `api2.yoob.com/api/v1/sessions/heartbeat` | Every 15 s while prepared | Session token |
| `api2.yoob.com/api/v1/sessions/end` | `destroy()` or page close | Session token |

With `YoobConversation`, microphone audio goes directly from the browser to OpenAI, and with `YoobGeminiConversation`
directly to Google. With `YoobLiveKitSession`, it goes to your LiveKit server, and the agent's audio comes back from it.

## Content Security Policy

Allow `connect-src https://cdn.yoob.com https://api2.yoob.com` (plus `wss://api.openai.com` or
`wss://generativelanguage.googleapis.com` for conversations, or your LiveKit server for LiveKit agents),
`script-src 'self' 'wasm-unsafe-eval'`, `worker-src 'self'`, and `img-src blob:` plus `media-src blob:` (the poster
and idle video are shown from verified in-memory copies). Workers and audio worklets ship as files, so scripts need no
`data:` or `blob:` source.

## License

Apache-2.0. Character model files are licensed separately. Includes ONNX Runtime Web (MIT); see NOTICE.
