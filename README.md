# Yoob Web SDK

Talking characters rendered in the browser. This repository holds:

| Path | What |
|---|---|
| [`packages/avatar`](packages/avatar) | `@yoob/avatar`: the SDK, with rendering, microphone and OpenAI Realtime conversations |
| [`apps/demo`](apps/demo) | A one-page demo: talk to Luna, pick a microphone, mute, see captions |
| [`examples/token-server`](examples/token-server) | The backend call that turns your Yoob API key into a browser session |

Start with the [package README](packages/avatar/README.md). For iOS, see
[Tap-Mobile/yoob-ios-sdk](https://github.com/Tap-Mobile/yoob-ios-sdk).

## Develop

```sh
npm install
npm run build
npm test
npm run demo        # http://localhost:5173 (expects a session endpoint on :3100)
```

## License

Apache-2.0. Character model files are licensed separately and are not in this repository.
