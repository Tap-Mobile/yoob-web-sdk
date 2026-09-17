import {
  decompressContoursScntz, decompressCropsScpk, decompressRefSref,
} from "./bank-codec";
import { assertLength, isLittleEndian } from "./binary-reader";
import { RuntimeAssetStore } from "./runtime-store";

export const RES = 320;
export const PLANE = RES * RES;
export const N_REFS = 30;
export const SERVE320_BUNDLE_ASSET_PATHS = [
  "bundle/meta.json",
  "bundle/pca.json",
  "bundle/ref_codebook.json",
  "bundle/idle_crops320.bin",
  "bundle/idle_anchors320.bin",
  "bundle/idle_stab_boxes.bin",
  "bundle/idle_contours320.bin",
  "bundle/idle_mouth_centers.bin",
  "bundle/idle_host_lip_y320.bin",
  "bundle/support320.bin",
  "bundle/input_mask320.bin",
  "bundle/ref_crops320.bin",
  "bundle/ref_anchors320.bin",
  "bundle/ref_geom6.bin",
] as const;

/** Packed bank siblings (optional; see tools/pack/stage_compact_web_runtime.py). */
export const SERVE320_PACKED_BANK_PATHS = [
  "bundle/idle_contours320.scntz",
  "bundle/idle_crops320.scpk",
  "bundle/ref_crops320.scntz",
] as const;

export interface Serve320Pca {
  geometry_mean: number[];
  components: number[][];
  score_std: number[];
}

export interface CodebookRef { frame: number; row: number }
export interface Serve320Codebook {
  strata: Record<"closed" | "mid" | "wide", number[]>;
  refs: CodebookRef[];
}

interface Serve320Meta {
  version: string;
  fps: number;
  n_frames: { idle: number; features: number };
  byte_order: string;
  /** Annie-style packed banks (decode-on-load). */
  pack?: {
    version: number;
    contours?: string;
    crops?: string;
    ref_crops?: string;
    drop_raw_banks?: boolean;
    /**
     * Hex SHA-256 of each *decoded* bank.
     *
     * The runtime manifest pins the digest of the packed file only, so without
     * this a decoder bug or a mis-staged pack reaches the compositor with
     * nothing but a length check behind it. Optional for back-compat; enforced
     * whenever present.
     */
    decoded_sha256?: {
      contours?: string;
      crops?: string;
      ref_crops?: string;
    };
  };
}

/** Hex SHA-256, matching the runtime manifest's digest format. */
async function sha256Hex(data: Uint8Array): Promise<string> {
  const view = data.buffer.slice(
    data.byteOffset, data.byteOffset + data.byteLength,
  ) as ArrayBuffer;
  const digest = await crypto.subtle.digest("SHA-256", view);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function assertDecodedDigest(
  label: string,
  data: Uint8Array,
  expected: string | undefined,
): Promise<void> {
  if (!expected) return;
  const digest = await sha256Hex(data);
  if (digest !== expected) {
    throw new Error(`${label}: sha256 ${digest} != ${expected}`);
  }
}

/**
 * Adopt a decoder result as an ArrayBuffer without a defensive copy when it
 * already owns its buffer exactly — avoids transiently doubling a 46 MB bank.
 */
export function ownedArrayBuffer(raw: Uint8Array): ArrayBuffer {
  if (raw.byteOffset === 0 && raw.byteLength === raw.buffer.byteLength) {
    return raw.buffer as ArrayBuffer;
  }
  return raw.buffer.slice(
    raw.byteOffset, raw.byteOffset + raw.byteLength,
  ) as ArrayBuffer;
}

export interface LoadProgress {
  name: string;
  loaded: number;
  total: number;
}

/**
 * The idle loop is a palindrome: it is built forward then reversed so the loop
 * is seamless, which makes rows `n/2+1 .. n-1` byte-identical mirrors of rows
 * `n/2-1 .. 1`. Measured on the shipped bank (150 frames): every idle table is
 * an exact mirror, and `concat(rows[0..75], rows[74..1])` reproduces
 * `idle_crops320.bin` with a matching SHA-256.
 *
 * brotli cannot exploit it — the mirror sits up to 46 MB away, past the 16 MB
 * window, and in reverse order — so half of the 14.36 MB wire cost of the crop
 * bank is duplicate bytes. A producer that stores only the unique prefix
 * (`web/tools/mirror-idle-crops.mjs`) therefore cuts 6.45 MB off the wire with
 * bit-exact reconstruction.
 *
 * Frames stored by such a producer, for `n` idle frames.
 */
export function mirroredPrefixFrames(nIdle: number): number {
  return nIdle / 2 + 1;
}

/**
 * Map a loop frame index onto the stored row.
 *
 * `storedFrames === nIdle` is the full bank and is the identity. Anything else
 * must be exactly the mirrored prefix, and indices past it fold back.
 */
export function mirroredIdleRow(
  index: number,
  storedFrames: number,
  nIdle: number,
): number {
  if (storedFrames === nIdle) return index;
  return index >= storedFrames ? nIdle - index : index;
}

/**
 * How many frames a crop bank of `byteLength` holds, or throw.
 *
 * Only the two legal shapes are accepted, so a truncated or mis-staged bank
 * cannot be silently reinterpreted as the other layout. The runtime manifest
 * already pins the exact byte count and SHA-256 of whichever one is published;
 * this is the second, in-runtime check.
 */
export function idleCropFrameCount(
  byteLength: number,
  nIdle: number,
  frameBytes: number,
): number {
  if (byteLength === nIdle * frameBytes) return nIdle;
  if (nIdle % 2 === 0) {
    const prefix = mirroredPrefixFrames(nIdle);
    if (byteLength === prefix * frameBytes) return prefix;
  }
  throw new Error(
    `idle_crops320.bin: ${byteLength} bytes is neither ${nIdle} frames nor a `
      + `mirrored ${nIdle % 2 === 0 ? mirroredPrefixFrames(nIdle) : "n/a"}-frame prefix`,
  );
}

async function mapLimit<T, R>(
  values: T[],
  limit: number,
  operation: (value: T) => Promise<R>,
): Promise<R[]> {
  const result = new Array<R>(values.length);
  let next = 0;
  async function worker() {
    while (next < values.length) {
      const index = next;
      next += 1;
      result[index] = await operation(values[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, values.length) }, () => worker()));
  return result;
}

export class Serve320Bundle {
  readonly nIdle: number;
  /** Rows physically present in `idleCrops`; see `mirroredIdleRow`. */
  readonly idleCropFrames: number;
  readonly meta: Serve320Meta;
  readonly pca: Serve320Pca;
  readonly codebook: Serve320Codebook;
  readonly idleCrops: Uint8Array;
  readonly idleAnchors: Float32Array;
  readonly stabBoxes: Int32Array;
  readonly idleContours: Uint8Array;
  readonly idleMouthCenters: Float32Array;
  readonly hostLipY: Float32Array;
  readonly support: Float32Array;
  readonly hole: Float32Array;
  readonly refCrops: Uint8Array;
  readonly refAnchors: Float32Array;
  readonly refGeom6: Float32Array;

  private constructor(values: {
    meta: Serve320Meta;
    pca: Serve320Pca;
    codebook: Serve320Codebook;
    idleCrops: ArrayBuffer;
    idleAnchors: ArrayBuffer;
    stabBoxes: ArrayBuffer;
    idleContours: ArrayBuffer;
    idleMouthCenters: ArrayBuffer;
    hostLipY: ArrayBuffer;
    support: ArrayBuffer;
    hole: ArrayBuffer;
    refCrops: ArrayBuffer;
    refAnchors: ArrayBuffer;
    refGeom6: ArrayBuffer;
  }) {
    this.meta = values.meta;
    this.nIdle = values.meta.n_frames.idle;
    this.idleCropFrames = idleCropFrameCount(
      values.idleCrops.byteLength, this.nIdle, 3 * PLANE,
    );
    this.pca = values.pca;
    this.codebook = values.codebook;
    this.idleCrops = new Uint8Array(values.idleCrops);
    this.idleAnchors = new Float32Array(values.idleAnchors);
    this.stabBoxes = new Int32Array(values.stabBoxes);
    this.idleContours = new Uint8Array(values.idleContours);
    this.idleMouthCenters = new Float32Array(values.idleMouthCenters);
    this.hostLipY = new Float32Array(values.hostLipY);
    this.support = new Float32Array(values.support);
    this.hole = new Float32Array(values.hole);
    this.refCrops = new Uint8Array(values.refCrops);
    this.refAnchors = new Float32Array(values.refAnchors);
    this.refGeom6 = new Float32Array(values.refGeom6);
  }

  static async load(
    store: RuntimeAssetStore,
    progress?: (value: LoadProgress) => void,
  ): Promise<Serve320Bundle> {
    if (!isLittleEndian()) throw new Error("Serve320 bundle requires a little-endian browser");
    const [meta, pca, codebook] = await Promise.all([
      store.json<Serve320Meta>("bundle/meta.json"),
      store.json<Serve320Pca>("bundle/pca.json"),
      store.json<Serve320Codebook>("bundle/ref_codebook.json"),
    ]);
    if (meta.n_frames.idle <= 0 || meta.n_frames.features !== 500) {
      throw new Error(`unsupported frame counts ${JSON.stringify(meta.n_frames)}`);
    }
    if (meta.byte_order !== "little-endian") throw new Error(`unsupported ${meta.byte_order}`);
    if (pca.geometry_mean.length !== 40 || pca.components.length !== 6 ||
        pca.components.some((row) => row.length !== 40) || pca.score_std.length !== 6) {
      throw new Error("pca.json shape mismatch");
    }
    const n = meta.n_frames.idle;
    const pack = meta.pack;
    // Small tables first so concurrent model fetches (in the pipeline worker)
    // compete less with multi‑MB crop/contour downloads on cold start.
    const files: Array<[string, number | null]> = [
      ["idle_anchors320.bin", n * 4 * 4],
      ["idle_stab_boxes.bin", n * 4 * 4],
      ["idle_mouth_centers.bin", n * 2 * 4],
      ["idle_host_lip_y320.bin", n * 4],
      ["support320.bin", PLANE * 4],
      ["input_mask320.bin", PLANE * 4],
      ["ref_anchors320.bin", N_REFS * 4 * 4],
      ["ref_geom6.bin", N_REFS * 6 * 4],
    ];
    if (pack?.ref_crops) files.push([pack.ref_crops, null]);
    else files.push(["ref_crops320.bin", N_REFS * 3 * PLANE]);
    if (pack?.contours) files.push([pack.contours, null]);
    else files.push(["idle_contours320.bin", n * PLANE]);
    if (pack?.crops) files.push([pack.crops, null]);
    // Either the full bank or its mirrored prefix is legal, so the exact byte
    // count is checked by `idleCropFrameCount` once the length is known — on
    // top of the manifest's own pinned `bytes` + SHA-256 for this path.
    else files.push(["idle_crops320.bin", null]);

    const loaded = await mapLimit(files, 3, async ([name, expected]) => {
      const data = await store.bytes(`bundle/${name}`);
      progress?.({ name, loaded: data.byteLength, total: expected ?? data.byteLength });
      if (expected !== null) assertLength(name, data.byteLength, expected);
      return data;
    });
    const byName = Object.fromEntries(
      files.map(([name], index) => [name, loaded[index]]),
    ) as Record<string, ArrayBuffer>;

    let idleContours: ArrayBuffer;
    let idleCrops: ArrayBuffer;
    let refCrops: ArrayBuffer;
    if (pack?.contours) {
      const raw = await decompressContoursScntz(new Uint8Array(byName[pack.contours]));
      assertLength("idle_contours (decoded)", raw.byteLength, n * PLANE);
      await assertDecodedDigest(
        "idle_contours (decoded)", raw, pack.decoded_sha256?.contours,
      );
      idleContours = ownedArrayBuffer(raw);
    } else {
      idleContours = byName["idle_contours320.bin"];
    }
    if (pack?.crops) {
      const raw = await decompressCropsScpk(new Uint8Array(byName[pack.crops]));
      // The packed bank carries whatever the raw bank carries — which since
      // the mirroring change is the 76-frame PREFIX, not all 150. This used to
      // assert the full length and rejected every mirror-packed bank with
      // `idle_crops (decoded): 23347200 bytes != 46080000`, while the
      // unpacked path beside it already accepted both shapes. Use the same
      // check, so packing and mirroring compose instead of colliding.
      idleCropFrameCount(raw.byteLength, n, 3 * PLANE);
      await assertDecodedDigest(
        "idle_crops (decoded)", raw, pack.decoded_sha256?.crops,
      );
      idleCrops = ownedArrayBuffer(raw);
    } else {
      idleCrops = byName["idle_crops320.bin"];
    }
    if (pack?.ref_crops) {
      const raw = await decompressRefSref(new Uint8Array(byName[pack.ref_crops]));
      assertLength("ref_crops (decoded)", raw.byteLength, N_REFS * 3 * PLANE);
      await assertDecodedDigest(
        "ref_crops (decoded)", raw, pack.decoded_sha256?.ref_crops,
      );
      refCrops = ownedArrayBuffer(raw);
    } else {
      refCrops = byName["ref_crops320.bin"];
    }

    return new Serve320Bundle({
      meta, pca, codebook,
      idleCrops,
      idleAnchors: byName["idle_anchors320.bin"],
      stabBoxes: byName["idle_stab_boxes.bin"],
      idleContours,
      idleMouthCenters: byName["idle_mouth_centers.bin"],
      hostLipY: byName["idle_host_lip_y320.bin"],
      support: byName["support320.bin"],
      hole: byName["input_mask320.bin"],
      refCrops,
      refAnchors: byName["ref_anchors320.bin"],
      refGeom6: byName["ref_geom6.bin"],
    });
  }

  idleCrop(index: number): Uint8Array {
    const row = mirroredIdleRow(index, this.idleCropFrames, this.nIdle);
    const start = row * 3 * PLANE;
    return this.idleCrops.subarray(start, start + 3 * PLANE);
  }

  contour(index: number): Uint8Array {
    const start = index * PLANE;
    return this.idleContours.subarray(start, start + PLANE);
  }

  anchor(index: number): [number, number, number, number] {
    const start = index * 4;
    return [this.idleAnchors[start], this.idleAnchors[start + 1],
      this.idleAnchors[start + 2], this.idleAnchors[start + 3]];
  }

  refAnchor(row: number): [number, number, number, number] {
    const start = row * 4;
    return [this.refAnchors[start], this.refAnchors[start + 1],
      this.refAnchors[start + 2], this.refAnchors[start + 3]];
  }

  refGeometry(row: number): Float32Array {
    return this.refGeom6.subarray(row * 6, row * 6 + 6);
  }

  refCrop(row: number): Uint8Array {
    const start = row * 3 * PLANE;
    return this.refCrops.subarray(start, start + 3 * PLANE);
  }

  mouthCenter(index: number): [number, number] {
    return [this.idleMouthCenters[index * 2], this.idleMouthCenters[index * 2 + 1]];
  }

  box(index: number): [number, number, number, number] {
    const start = index * 4;
    return [this.stabBoxes[start], this.stabBoxes[start + 1],
      this.stabBoxes[start + 2], this.stabBoxes[start + 3]];
  }
}
