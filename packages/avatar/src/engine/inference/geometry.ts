import { createSession, ort } from "./ort-runtime";

/**
 * Geometry feature width.
 *
 * The exported model used to take 3072 = 1024 audio + TWO IDENTICAL 1024-wide
 * phoneme one-hots (this file wrote the same id at 1024+id and at 2048+id), with
 * the id clamped to <= 51. Of those 3072 columns, 1024 were a byte-exact
 * duplicate of another 1024 and 1944 could never be nonzero at all.
 *
 * Only `onnx::MatMul_470` touched that dimension, so the duplicate slots fold
 * algebraically:
 *
 *   y = audio·W[0:1024] + onehot·W[1024:2048] + onehot·W[2048:3072]
 *     = audio·W[0:1024] + onehot·(W[1024:2048] + W[2048:3072])
 *
 * leaving 1024 audio + 52 reachable phone classes. Verified against the old
 * graph on 600 corpus frames: contact agreement 100.0%, landmark p95 7.6e-05,
 * pred6 relative RMS 0.126% — pure fp16 rounding, since the fold sums the two
 * weight rows in fp32 and rounds once instead of rounding two products.
 * Removes 1,532,930 bytes of model weights.
 */
const AUDIO_DIM = 1024;
const PHONE_CLASSES = 52;
export const GEOMETRY_FEATURE_DIM = AUDIO_DIM + PHONE_CLASSES;

export interface GeometryOutput {
  pred6: Float32Array;
  contactLogit: Float32Array;
  frameCount: number;
}

export function assembleGeometryFeatures(
  audioRows: Float32Array,
  silenceRows: Float32Array,
  phoneIDs: Uint8Array,
  frameCount: number,
  destination?: Float32Array,
  tailFrames = 25,
): Float32Array {
  const total = frameCount + tailFrames;
  if (frameCount < 1 || total > 750) throw new Error(`unsupported geometry length ${frameCount}`);
  if (audioRows.length !== frameCount * 1024 ||
      silenceRows.length < tailFrames * 1024 ||
      phoneIDs.length !== frameCount) {
    throw new Error("geometry feature shape mismatch");
  }
  const needed = total * GEOMETRY_FEATURE_DIM;
  const features = destination && destination.length >= needed
    ? destination
    : new Float32Array(needed);
  // Reused buffers retain previous one-hots; clear the active window.
  features.fill(0, 0, needed);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const dest = frame * GEOMETRY_FEATURE_DIM;
    features.set(audioRows.subarray(frame * AUDIO_DIM, (frame + 1) * AUDIO_DIM), dest);
    const phoneID = Math.max(0, Math.min(PHONE_CLASSES - 1, phoneIDs[frame]));
    features[dest + AUDIO_DIM + phoneID] = 1;
  }
  for (let frame = 0; frame < tailFrames; frame += 1) {
    const dest = (frameCount + frame) * GEOMETRY_FEATURE_DIM;
    features.set(silenceRows.subarray(frame * AUDIO_DIM, (frame + 1) * AUDIO_DIM), dest);
    features[dest + AUDIO_DIM] = 1;   // silence class
  }
  return features.subarray(0, needed);
}

export class GeometryRuntime {
  private features = new Float32Array(0);
  private featuresCapacity = 0;
  private pred6 = new Float32Array(0);
  private contactLogit = new Float32Array(0);

  private constructor(private readonly session: ort.InferenceSession) {}

  static async load(model: string | Uint8Array) {
    // The dynamic BiGRU is deliberately WASM-only for the correctness MVP.
    // Renderer and Feather own the WebGPU lane; placement is explicit rather
    // than silently depending on ORT partitioning.
    return new GeometryRuntime(await createSession(
      model, ["wasm"], "Serve320 geometry",
    ));
  }

  private ensureFeatures(total: number): Float32Array {
    const needed = total * GEOMETRY_FEATURE_DIM;
    if (needed > this.featuresCapacity) {
      this.features = new Float32Array(needed);
      this.featuresCapacity = needed;
    }
    return this.features;
  }

  async run(
    audioRows: Float32Array,
    silenceRows: Float32Array,
    phoneIDs: Uint8Array,
    frameCount: number,
  ): Promise<GeometryOutput> {
    return this.runWindow(audioRows, silenceRows, phoneIDs, frameCount, true);
  }

  async runWindow(
    audioRows: Float32Array,
    silenceRows: Float32Array,
    phoneIDs: Uint8Array,
    frameCount: number,
    final: boolean,
  ): Promise<GeometryOutput> {
    const tailFrames = final ? 25 : 0;
    const total = frameCount + tailFrames;
    const features = assembleGeometryFeatures(
      audioRows, silenceRows, phoneIDs, frameCount, this.ensureFeatures(total), tailFrames,
    );
    // One tensor per utterance (not per frame). Shape is dynamic.
    const featsTensor = new ort.Tensor("float32", features, [1, total, GEOMETRY_FEATURE_DIM]);
    const result = await this.session.run({ feats: featsTensor });
    const predTensor = result.pred6;
    const contactTensor = result.contact_logit;
    if (!predTensor || !contactTensor || predTensor.dims[1] !== total ||
        contactTensor.dims[1] !== total) throw new Error("bad geometry outputs");
    const predAll = predTensor.data as Float32Array;
    const contactAll = contactTensor.data as Float32Array;
    if (this.pred6.length < frameCount * 6) this.pred6 = new Float32Array(frameCount * 6);
    if (this.contactLogit.length < frameCount) this.contactLogit = new Float32Array(frameCount);
    this.pred6.set(predAll.subarray(0, frameCount * 6));
    this.contactLogit.set(contactAll.subarray(0, frameCount));
    predTensor.dispose();
    contactTensor.dispose();
    featsTensor.dispose();
    return {
      pred6: this.pred6.subarray(0, frameCount * 6),
      contactLogit: this.contactLogit.subarray(0, frameCount),
      frameCount,
    };
  }

  async warmup(): Promise<void> {
    await this.run(
      new Float32Array(1024),
      new Float32Array(25 * 1024),
      new Uint8Array(1),
      1,
    );
  }
}
