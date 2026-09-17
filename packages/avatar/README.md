# @yoob/avatar

Talking characters rendered in the browser with WebGPU. You give Yoob speech; it plays the audio and moves the face in
sync, on the visitor's own GPU. Add a microphone and OpenAI Realtime, and you have a live spoken conversation with
barge-in.

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

With `YoobConversation`, microphone audio goes directly from the browser to OpenAI.

## Content Security Policy

Allow `connect-src https://cdn.yoob.com https://api2.yoob.com` (plus `wss://api.openai.com` for conversations),
`script-src 'self' 'wasm-unsafe-eval'`, `worker-src 'self'`, and `img-src blob:` plus `media-src blob:` (the poster
and idle video are shown from verified in-memory copies). Workers and audio worklets ship as files, so scripts need no
`data:` or `blob:` source.

## License

Apache-2.0. Character model files are licensed separately. Includes ONNX Runtime Web (MIT); see NOTICE.
