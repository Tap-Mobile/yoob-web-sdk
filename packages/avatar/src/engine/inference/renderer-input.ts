import { float32ToFloat16Bits } from "../math/rounding";

interface NativeFloat16ArrayLike {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly length: number;
  set(source: ArrayLike<number>): void;
}

export type NativeFloat16ArrayConstructor = new (
  length: number,
) => NativeFloat16ArrayLike;

export interface Float16InputStaging {
  /** Raw IEEE-754 binary16 bits accepted by ORT and WebGPU uploads. */
  readonly bits: Uint16Array;
  readonly usesNativeFloat16: boolean;
  write(source: Float32Array): Uint16Array;
}

function nativeFloat16ArrayConstructor(): NativeFloat16ArrayConstructor | null {
  return (globalThis as unknown as {
    Float16Array?: NativeFloat16ArrayConstructor;
  }).Float16Array ?? null;
}

/**
 * Reusable float32 -> float16 staging for the renderer input boundary.
 *
 * Current Chromium provides Float16Array, which performs the conversion in
 * native code. The explicit IEEE converter keeps the same descriptor usable
 * on older engines instead of silently loading a float16 model with float32
 * input. Both paths expose Uint16Array bits because that is ORT's portable
 * representation for a float16 tensor.
 */
export function createFloat16InputStaging(
  length: number,
  nativeConstructor: NativeFloat16ArrayConstructor | null = nativeFloat16ArrayConstructor(),
): Float16InputStaging {
  if (!Number.isSafeInteger(length) || length < 1) {
    throw new Error(`invalid float16 staging length ${length}`);
  }
  if (nativeConstructor) {
    const values = new nativeConstructor(length);
    const bits = new Uint16Array(values.buffer, values.byteOffset, values.length);
    return {
      bits,
      usesNativeFloat16: true,
      write(source) {
        if (source.length !== length) {
          throw new Error(`bad renderer input ${source.length}`);
        }
        values.set(source);
        return bits;
      },
    };
  }

  const bits = new Uint16Array(length);
  return {
    bits,
    usesNativeFloat16: false,
    write(source) {
      if (source.length !== length) {
        throw new Error(`bad renderer input ${source.length}`);
      }
      for (let index = 0; index < source.length; index += 1) {
        bits[index] = float32ToFloat16Bits(source[index]);
      }
      return bits;
    },
  };
}
