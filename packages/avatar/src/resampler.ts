/**
 * Streaming PCM16 sample-rate converter: a short windowed-sinc filter evaluated at each output position. It keeps its
 * history between packets, so a stream cut into 20 ms packets converts exactly as it would in one piece. When it
 * lowers the rate, the filter also removes what the new rate can't represent (24 kHz → 16 kHz cuts above ~7 kHz).
 */
export class PcmResampler {
  private readonly step: number;
  private readonly cutoff: number;
  private buffer: Float32Array;
  private position: number;

  constructor(readonly fromRate: number, readonly toRate: number, private readonly halfTaps = 8) {
    if (!(fromRate > 0) || !(toRate > 0)) throw new RangeError("Sample rates must be positive.");
    this.step = fromRate / toRate;
    this.cutoff = Math.min(1, toRate / fromRate) * 0.9;
    // Silent history lets the first output sample line up with the first input sample.
    this.buffer = new Float32Array(halfTaps);
    this.position = halfTaps;
  }

  process(input: Int16Array): Int16Array {
    if (this.fromRate === this.toRate) return input.slice();
    const buffer = new Float32Array(this.buffer.length + input.length);
    buffer.set(this.buffer);
    buffer.set(input, this.buffer.length);
    const taps = this.halfTaps;
    const output: number[] = [];
    let position = this.position;
    while (Math.floor(position) + taps < buffer.length) {
      const base = Math.floor(position);
      let sum = 0;
      let weight = 0;
      for (let k = base - taps + 1; k <= base + taps; k += 1) {
        const w = this.kernel(position - k);
        sum += buffer[k] * w;
        weight += w;
      }
      const value = Math.round(weight ? sum / weight : 0);
      output.push(value > 32767 ? 32767 : value < -32768 ? -32768 : value);
      position += this.step;
    }
    const drop = Math.max(0, Math.floor(position) - taps + 1);
    this.buffer = buffer.slice(drop);
    this.position = position - drop;
    return Int16Array.from(output);
  }

  private kernel(distance: number): number {
    if (Math.abs(distance) >= this.halfTaps) return 0;
    const x = Math.PI * this.cutoff * distance;
    const sinc = x === 0 ? 1 : Math.sin(x) / x;
    const window = 0.5 * (1 + Math.cos((Math.PI * distance) / this.halfTaps));
    return sinc * window;
  }
}
