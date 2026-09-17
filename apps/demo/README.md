# Yoob web demo

Talk to Luna in the browser: microphone choice, mute, level meter, captions, and a sample line.

```sh
npm install && npm run build   # at the repository root
YOOB_API_KEY=… node ../../examples/token-server/server.mjs   # serves /yoob-session on :3100
npm run demo
```

For spoken conversations, add an `/openai-secret` route to the token server that returns an OpenAI Realtime client
secret (`{ "value": "ek_…" }`).
