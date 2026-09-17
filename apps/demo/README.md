# Yoob web demo

Talk to Luna in the browser: microphone choice, mute, level meter, captions, and a sample line.

```sh
npm install && npm run build   # at the repository root
YOOB_API_KEY=… node ../../examples/token-server/server.mjs   # serves /yoob-session and /yoob-voice on :3100
npm run demo
```

Talk uses Yoob voice, so no provider key is needed; minutes are billed to your Yoob workspace. To use your own OpenAI
account instead, start the token server with `OPENAI_API_KEY` and switch `getVoiceSession` in `src/main.ts` to the
`getClientSecret` line next to it.
