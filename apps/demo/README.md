# Yoob web demo

Talk to Luna in the browser: microphone choice, mute, level meter, captions, and a sample line.

```sh
npm install && npm run build   # at the repository root
YOOB_API_KEY=yoob_test_… npm run token-server   # at the root: /yoob-session and /yoob-voice on 127.0.0.1:3100
npm run demo
```

`npm run token-server` sets `YOOB_EXAMPLE_ALLOW_ANONYMOUS=1`, which lets anyone who can reach the server mint
sessions. That is fine on your machine only; before you deploy a token server, replace `requireUser()` with your own
sign-in check. A `yoob_test_` key opens sandbox sessions that don't use credits.

Talk uses Yoob voice, so no provider key is needed; minutes are billed to your Yoob workspace. To use your own OpenAI
account instead, start the token server with `OPENAI_API_KEY` and switch `getVoiceSession` in `src/main.ts` to the
`getClientSecret` line next to it.
