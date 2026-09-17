/**
 * Decode CallAnnie-style packed Serve320 banks
 * (see tools/pack/compress_serve320_banks.py).
 * Contours: SCNT + zlib. Crops: SCPK keyframe/delta + zlib.
 */

const TE = new TextDecoder();

function u32(view: DataView, offset: number): number {
  return view.getUint32(offset, true);
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    const zlib = await import("node:zlib");
    return new Promise((resolve, reject) => {
      zlib.inflate(Buffer.from(data), (error, result) => {
        if (error) reject(error);
        else resolve(new Uint8Array(result));
      });
    });
  }
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);
  const stream = new Blob([copy]).stream().pipeThrough(
    new DecompressionStream("deflate"),
  );
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

export async function decompressContoursScntz(packed: Uint8Array): Promise<Uint8Array> {
  const magic = TE.decode(packed.subarray(0, 4));
  if (magic !== "SCNT") throw new Error(`bad contour magic ${magic}`);
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const n = u32(view, 4);
  const len = u32(view, 8);
  const raw = await inflate(packed.subarray(12, 12 + len));
  if (raw.byteLength !== n * 320 * 320) {
    throw new Error(`contour inflate size ${raw.byteLength}`);
  }
  return raw;
}

export async function decompressRefSref(packed: Uint8Array): Promise<Uint8Array> {
  const magic = TE.decode(packed.subarray(0, 4));
  if (magic !== "SREF") throw new Error(`bad ref magic ${magic}`);
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const len = u32(view, 4);
  return inflate(packed.subarray(8, 8 + len));
}

export async function decompressCropsScpk(packed: Uint8Array): Promise<Uint8Array> {
  const magic = TE.decode(packed.subarray(0, 4));
  if (magic !== "SCPK") throw new Error(`bad crops magic ${magic}`);
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const n = u32(view, 4);
  const plane3 = u32(view, 12);
  let offset = 16;
  const out = new Uint8Array(n * plane3);
  let prev: Uint8Array | undefined;
  for (let i = 0; i < n; i += 1) {
    const kind = packed[offset];
    const len = u32(view, offset + 1);
    offset += 5;
    const blob = packed.subarray(offset, offset + len);
    offset += len;
    const decoded = await inflate(blob);
    const frame = out.subarray(i * plane3, (i + 1) * plane3);
    if (kind === 1) {
      if (decoded.byteLength !== plane3) throw new Error(`key crop ${i}`);
      frame.set(decoded);
    } else {
      // int16 little-endian deltas (2 bytes per plane sample)
      if (!prev || decoded.byteLength !== plane3 * 2) throw new Error(`delta crop ${i}`);
      const deltas = new Int16Array(
        decoded.buffer, decoded.byteOffset, plane3,
      );
      for (let p = 0; p < plane3; p += 1) {
        frame[p] = Math.max(0, Math.min(255, prev[p] + deltas[p]));
      }
    }
    prev = frame.slice();
  }
  return out;
}
