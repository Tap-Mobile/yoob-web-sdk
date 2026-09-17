# Changelog

## 0.2.0

Security hardening. Needs the Yoob API that ships with it (heartbeats with renewed grants).

- **Heartbeats are enforced.** They start when `prepare()` opens the session, not after the download. The character
  stops rendering, the phase becomes `stopped`, and the new `onSessionEnded` option fires when Yoob refuses the session
  (401/403 → `unauthorized`), the workspace is out of credit (402 or `stop` with `out-of-credits` → `out-of-credit`),
  or three heartbeats in a row fail (new error code `session-ended`). Transient failures are retried with backoff
  within those three attempts. A session Yoob no longer knows (404, or `stop` for another reason) is replaced through
  `getCredentials()`.
- **Renewed download grants.** A heartbeat reply may carry `download_token` (with `download_token_expires_at`; the
  names `grant` and `grant_expires_at` are also accepted); the page and the render worker use it for the next
  downloads. Replies without it work as before.
- **Terminal stop reasons.** `stop` with `sandbox-limit` or `suspended` ends the session (`session-ended`), and
  `key-revoked` ends it as `unauthorized`, instead of opening a new session.
- **Voice host pinning.** Yoob voice sessions connect only to `wss://*.yoob.com`. The new `voiceHosts` option allows
  a self-hosted relay.
- **Credentials check.** `getCredentials()` must return a `session_token` and a `download_token`; anything else fails
  with `unauthorized` and a clear message.
- **Example token server.** Fails closed until you add your own auth (`YOOB_EXAMPLE_ALLOW_ANONYMOUS=1` for local
  development, set by `npm run token-server`), accepts only characters in `YOOB_CHARACTERS`
  (default `luna-realistic,luna-anime`) and never asks for `*`, rate-limits each user, always pins the voice prompt,
  and never passes request fields such as `is_sandbox` through. Sandbox sessions come from `yoob_test_` keys.
- README: new Security section.

## 0.1.1

- Install fix: no postinstall step, no runtime dependencies.

## 0.1.0

- First release: WebGPU characters, Yoob voice, your own OpenAI Realtime, Gemini Live and LiveKit agents.
