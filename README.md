# Yoob Web SDK

Talking characters rendered in the browser. This repository holds:

| Path | What |
|---|---|
| [`packages/avatar`](packages/avatar) | `@yoob/avatar`: the SDK, with rendering, microphone, and conversations with Yoob voice, OpenAI Realtime, Gemini Live or LiveKit |
| [`apps/demo`](apps/demo) | A one-page demo: talk to Luna, pick a microphone, mute, see captions |
| [`examples/token-server`](examples/token-server) | The backend calls that turn your Yoob API key into a browser session and a voice session |

Start with the [package README](packages/avatar/README.md). For iOS, see
[Yoob-com/yoob-ios-sdk](https://github.com/Yoob-com/yoob-ios-sdk).

## Develop

```sh
npm install
npm run build
npm test
npm run demo        # http://localhost:5173 (expects a session endpoint on :3100)
```

## License

Apache-2.0. Character model files are licensed separately and are not in this repository.
