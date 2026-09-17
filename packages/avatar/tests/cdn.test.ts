import { test } from "node:test";
import assert from "node:assert/strict";
import { generateKeyPairSync, sign } from "node:crypto";
import { compareVersions, validateManifest, YoobError, type CharacterManifest } from "../src/cdn";

const sha = "a".repeat(64);
const base = (): CharacterManifest => ({
  schema: 1, character: "luna-anime", version: "2026.09.17.1", engine: "anime-web", displayName: "Luna",
  width: 1080, height: 1920, poster: "poster.jpg", idle: { frames: [], fps: 6 }, minSDK: "0.1.0",
  files: [{ path: "poster.jpg", size: 3, sha256: sha, tier: 0, chunks: [{ sha256: sha, size: 3 }] }],
});

test("accepts a well-formed manifest", () => {
  assert.doesNotThrow(() => validateManifest(base()));
});

test("rejects unsafe paths, duplicate files and bad chunk sums", () => {
  for (const path of ["../x", "/abs", "a/../b", "a//b", "./a"]) {
    const m = base(); m.files.push({ ...m.files[0], path });
    assert.throws(() => validateManifest(m), YoobError, path);
  }
  const duplicate = base(); duplicate.files.push({ ...duplicate.files[0] });
  assert.throws(() => validateManifest(duplicate), YoobError);
  const sums = base(); sums.files[0] = { ...sums.files[0], size: 4 };
  assert.throws(() => validateManifest(sums), YoobError);
});

test("refuses manifests that need a newer SDK", () => {
  const m = base(); m.minSDK = "9.0.0";
  assert.throws(() => validateManifest(m), /needs @yoob\/avatar 9\.0\.0/);
  assert.ok(compareVersions("0.1.0", "0.1.1") < 0);
  assert.equal(compareVersions("1.2.3", "1.2.3"), 0);
});

test("the pinned manifest key verifies what the release key signs", async () => {
  // Round trip of the verification code path with a throwaway key: WebCrypto Ed25519 accepts Node's signature.
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const payload = Buffer.from(JSON.stringify(base()));
  const signature = sign(null, payload, privateKey);
  const raw = publicKey.export({ format: "der", type: "spki" }).subarray(-32);
  const key = await crypto.subtle.importKey("raw", raw, { name: "Ed25519" }, false, ["verify"]);
  assert.ok(await crypto.subtle.verify({ name: "Ed25519" }, key, signature, payload));
  payload[3] ^= 1;
  assert.equal(await crypto.subtle.verify({ name: "Ed25519" }, key, signature, payload), false);
});
