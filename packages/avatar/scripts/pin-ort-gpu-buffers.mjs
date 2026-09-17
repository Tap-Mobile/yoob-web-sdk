/**
 * Pin caller-owned external GPU buffers in onnxruntime-web.
 *
 * ORT 1.27's WebGPU EP is native C++ compiled via emdawnwebgpu. A GPUBuffer
 * handle IS the WASM heap address of the C++ WGPUBufferImpl, and `run()`'s
 * `finally` calls webgpuUnregisterBuffer on every external input/output tensor
 * on EVERY run. Registration is once, unregistration is per-run, so the
 * refcount reaches 0 after the first run, `_wgpuBufferRelease` frees the impl
 * back into the shared dlmalloc heap, and the slot in the handle table is
 * deleted.
 *
 * With graph capture the recorded bind group still references that handle. On
 * replay the lookup returns undefined and Chrome throws:
 *
 *   Failed to execute 'createBindGroup' ... Failed to read the 'buffer'
 *   property from 'GPUBufferBinding': Required member is undefined.
 *
 * Whether it detonates depends on whether the freed address is recycled before
 * the replay, which depends on the whole allocation history of the single
 * shared heap — which is why an unrelated, smaller geometry model flipped it
 * deterministically, and why an isolated repro could not reproduce it.
 *
 * This patch skips release-on-zero for buffers flagged `__ortPin`, giving them
 * the session-lifetime that ORT already grants its own EP-owned buffers.
 * src/inference/renderer.ts sets the flag before the first run.
 *
 * Idempotent. Run from postinstall so `npm ci` cannot silently drop it.
 */
import { readFileSync, writeFileSync, copyFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { createRequire } from "node:module";
import { dirname as dirOf } from "node:path";
const dist = (() => { try { return dirOf(createRequire(import.meta.url).resolve("onnxruntime-web")); } catch { return "node_modules/onnxruntime-web/dist"; } })();
// <meta>[1]--, <meta>[1]===0 && ( <release>(<meta>[0]), delete <buffer>[<sym>] )
const pattern =
  /(?<m>[A-Za-z_$]+)\[1\]--,\k<m>\[1\]===0&&\((?<rel>[A-Za-z_$]+)\(\k<m>\[0\]\),delete (?<buf>[A-Za-z_$]+)\[(?<h>[A-Za-z_$]+)\]\)/;

if (!existsSync(dist)) {
  console.log("onnxruntime-web not installed; skipping GPU buffer pin patch");
  process.exit(0);
}
let patched = 0;
let already = 0;
for (const name of readdirSync(dist).filter((f) => f.endsWith(".mjs"))) {
  const path = join(dist, name);
  const source = readFileSync(path, "utf8");
  if (source.includes("__ortPin")) { already += 1; continue; }
  const match = pattern.exec(source);
  if (!match) continue;
  if (!existsSync(`${path}.orig`)) copyFileSync(path, `${path}.orig`);
  const replaced = match[0].replace("===0&&(", `===0&&!${match.groups.buf}.__ortPin&&(`);
  writeFileSync(path, source.replace(match[0], replaced));
  console.log(`  pinned external GPU buffers in ${name}`);
  patched += 1;
}
console.log(`ort GPU buffer pin: ${patched} patched, ${already} already patched`);
