import { test } from "node:test";
import assert from "node:assert/strict";
import { PcmResampler } from "../src/resampler";

const tone = (hz: number, rate: number, count: number, amplitude = 10_000) =>
  Int16Array.from({ length: count }, (_, i) => Math.round(amplitude * Math.sin((2 * Math.PI * hz * i) / rate)));

const rms = (samples: Int16Array) => Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0) / samples.length);

function streamed(resampler: PcmResampler, input: Int16Array, packet: number): Int16Array {
  const parts: number[] = [];
  for (let i = 0; i < input.length; i += packet) parts.push(...resampler.process(input.subarray(i, i + packet)));
  return Int16Array.from(parts);
}

test("24 kHz → 16 kHz keeps two samples in three and matches across packet boundaries", () => {
  const input = tone(440, 24_000, 24_000);
  const whole = new PcmResampler(24_000, 16_000).process(input);
  const packets = streamed(new PcmResampler(24_000, 16_000), input, 480);
  assert.deepEqual(packets, whole);
  assert.ok(Math.abs(whole.length - 16_000) <= 8, `got ${whole.length}`);
});

test("speech-band tones pass unchanged and stay in phase", () => {
  const output = streamed(new PcmResampler(24_000, 16_000), tone(440, 24_000, 24_000), 480);
  const expected = tone(440, 16_000, output.length);
  let worst = 0;
  for (let i = 32; i < output.length; i += 1) worst = Math.max(worst, Math.abs(output[i] - expected[i]));
  assert.ok(worst < 150, `worst error ${worst} of 10000`);
});

test("content above the new Nyquist is filtered instead of aliased", () => {
  const output = new PcmResampler(24_000, 16_000).process(tone(10_000, 24_000, 24_000));
  assert.ok(rms(output.subarray(64)) < 0.1 * rms(tone(10_000, 24_000, 24_000)), `rms ${rms(output)}`);
});

test("upsampling and equal rates", () => {
  const input = tone(300, 16_000, 1_600);
  const up = streamed(new PcmResampler(16_000, 24_000), input, 320);
  assert.ok(Math.abs(up.length - 2_400) <= 16, `got ${up.length}`);
  const same = new PcmResampler(24_000, 24_000).process(input);
  assert.deepEqual(same, input);
  assert.notEqual(same, input);
  const dc = new PcmResampler(24_000, 16_000).process(new Int16Array(480).fill(-32768));
  assert.equal(dc.at(-1), -32768);
});
