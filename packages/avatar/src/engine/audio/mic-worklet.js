const RESAMPLER_TABLES = new Map();

function concatenateFloat32(left, right) {
  const output = new Float32Array(left.length + right.length);
  output.set(left);
  output.set(right, left.length);
  return output;
}

function coefficientTable(sourceRate, targetRate, taps, phases) {
  const key = `${sourceRate}:${targetRate}:${taps}:${phases}`;
  const cached = RESAMPLER_TABLES.get(key);
  if (cached) return cached;
  const half = Math.floor(taps / 2);
  // Preserve the target Nyquist band with a small transition region. This is
  // the anti-alias filter the former linear 48→24 kHz path was missing.
  const cutoff = 0.5 * Math.min(1, targetRate / sourceRate) * 0.94;
  const table = new Float32Array(phases * taps);
  for (let phase = 0; phase < phases; phase += 1) {
    const fraction = phase / phases;
    let gain = 0;
    for (let tap = 0; tap < taps; tap += 1) {
      const offset = tap - half;
      const distance = offset - fraction;
      const angle = 2 * Math.PI * cutoff * distance;
      const sinc = Math.abs(angle) < 1e-9 ? 1 : Math.sin(angle) / angle;
      const window = 0.42
        + 0.5 * Math.cos(Math.PI * offset / half)
        + 0.08 * Math.cos(2 * Math.PI * offset / half);
      const value = 2 * cutoff * sinc * window;
      table[phase * taps + tap] = value;
      gain += value;
    }
    for (let tap = 0; tap < taps; tap += 1) {
      table[phase * taps + tap] /= gain;
    }
  }
  RESAMPLER_TABLES.set(key, table);
  return table;
}

/** Stateful windowed-sinc resampler shared by every microphone packet. */
export class BandlimitedResampler {
  constructor(sourceRate, targetRate, taps = 33, phases = 1024) {
    if (!(sourceRate > 0) || !(targetRate > 0) || taps < 5 || taps % 2 !== 1) {
      throw new Error("invalid band-limited resampler configuration");
    }
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.taps = taps;
    this.phases = phases;
    this.half = Math.floor(taps / 2);
    this.step = sourceRate / targetRate;
    this.table = coefficientTable(sourceRate, targetRate, taps, phases);
    this.buffer = new Float32Array(this.half);
    this.position = this.half;
    this.realSamples = 0;
    this.flushed = false;
  }

  push(input) {
    if (this.flushed) throw new Error("resampler was already flushed");
    if (input.length === 0) return new Float32Array();
    this.realSamples += input.length;
    this.buffer = concatenateFloat32(this.buffer, input);
    return this.consume(Number.POSITIVE_INFINITY);
  }

  flush() {
    if (this.flushed || this.realSamples === 0) return new Float32Array();
    this.flushed = true;
    const lastRealPosition = this.buffer.length - 1;
    this.buffer = concatenateFloat32(
      this.buffer,
      new Float32Array(this.half + Math.ceil(this.step) + 1),
    );
    return this.consume(lastRealPosition);
  }

  consume(maxPosition) {
    const output = [];
    while (
      this.position <= maxPosition
      && Math.floor(this.position) + this.half < this.buffer.length
    ) {
      const center = Math.floor(this.position);
      const fraction = this.position - center;
      const phase = Math.min(this.phases - 1, Math.floor(fraction * this.phases));
      const tableOffset = phase * this.taps;
      let value = 0;
      for (let tap = 0; tap < this.taps; tap += 1) {
        value += this.buffer[center + tap - this.half] * this.table[tableOffset + tap];
      }
      output.push(value);
      this.position += this.step;
    }
    const discard = Math.max(0, Math.floor(this.position) - this.half);
    if (discard > 0) {
      this.buffer = this.buffer.slice(discard);
      this.position -= discard;
    }
    return Float32Array.from(output);
  }
}

const WorkletProcessorBase = globalThis.AudioWorkletProcessor || class {};

class Serve320MicProcessor extends WorkletProcessorBase {
  constructor() {
    super();
    this.enabled = false;
    this.resampler = new BandlimitedResampler(sampleRate, 24000);
    // 20 ms packets keep VAD/barge-in responsive while preserving the native
    // HF/Qwen 24 kHz wire contract.
    this.chunk = new Int16Array(480);
    this.chunkLength = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "enabled") {
        this.enabled = Boolean(data.value);
        if (this.enabled) {
          this.resampler = new BandlimitedResampler(sampleRate, 24000);
          this.chunkLength = 0;
        }
      }
    };
  }

  emit(value) {
    const clamped = Math.max(-1, Math.min(1, value));
    this.chunk[this.chunkLength++] = clamped < 0
      ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    if (this.chunkLength === this.chunk.length) {
      const payload = this.chunk.buffer;
      this.port.postMessage({ type: "pcm", pcm: payload }, [payload]);
      this.chunk = new Int16Array(480);
      this.chunkLength = 0;
    }
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || !this.enabled) return true;
    const output = this.resampler.push(input);
    for (let index = 0; index < output.length; index += 1) this.emit(output[index]);
    return true;
  }
}

if (typeof registerProcessor === "function") {
  registerProcessor("serve320-mic", Serve320MicProcessor);
}
