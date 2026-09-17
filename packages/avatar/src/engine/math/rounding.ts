const scratch = new Float32Array(1);
const scratchBits = new Uint32Array(scratch.buffer);

export function roundToEven(value: number): number {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function float32ToFloat16Bits(value: number): number {
  scratch[0] = value;
  const bits = scratchBits[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (((bits >>> 23) & 0xff) === 0xff) {
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  }
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    const round = mantissa & 0x1fff;
    mantissa >>>= 13;
    if (round > 0x1000 || (round === 0x1000 && (mantissa & 1))) mantissa += 1;
    return sign | mantissa;
  }
  if (exponent >= 31) return sign | 0x7c00;
  const round = mantissa & 0x1fff;
  mantissa >>>= 13;
  if (round > 0x1000 || (round === 0x1000 && (mantissa & 1))) {
    mantissa += 1;
    if (mantissa === 0x400) {
      mantissa = 0;
      exponent += 1;
      if (exponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exponent << 10) | mantissa;
}

export function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x3ff;
  let out: number;
  if (exponent === 0) {
    if (mantissa === 0) out = sign;
    else {
      exponent = 1;
      while ((mantissa & 0x400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x3ff;
      out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent === 31) {
    out = sign | 0x7f800000 | (mantissa << 13);
  } else {
    out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  scratchBits[0] = out >>> 0;
  return scratch[0];
}

export function quantizeFloat16(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}
