import { defineConfig } from "vite";

// /yoob-session goes to your backend (Examples: ../../examples/token-server). Here it is proxied to port 3100.
export default defineConfig({
  server: { proxy: { "/yoob-session": "http://127.0.0.1:3100", "/yoob-voice": "http://127.0.0.1:3100", "/openai-secret": "http://127.0.0.1:3100" } },
  optimizeDeps: { exclude: ["@yoob/avatar"] },
});
