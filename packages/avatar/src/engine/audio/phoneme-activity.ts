import { roundToEven } from "../math/rounding";

export const SILENCE_PHONE_ID = 0;
export const FALLBACK_OPEN_PHONE_ID = 26;

export function frameRms(pcm: Float32Array, frameCount: number): Float32Array {
  const output = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * 640;
    const end = Math.min(start + 640, pcm.length);
    if (end <= start) continue;
    let sumSquares = 0;
    for (let sample = start; sample < end; sample += 1) {
      sumSquares += pcm[sample] * pcm[sample];
    }
    output[frame] = Math.sqrt(sumSquares / (end - start));
  }
  return output;
}

function percentile(sorted: Float32Array, probability: number): number {
  const index = Math.max(0, Math.min(
    sorted.length - 1,
    roundToEven((sorted.length - 1) * probability),
  ));
  return sorted[index];
}

export function silenceThreshold(rms: Float32Array): number {
  if (rms.length === 0) return 0;
  const maximum = Math.max(...rms);
  if (maximum <= 0) return 0;
  const sorted = Float32Array.from(rms).sort();
  const noiseFloor = Math.min(percentile(sorted, 0.05), percentile(sorted, 0.10));
  const adaptive = Math.max(noiseFloor * 4, maximum * 0.010);
  const capped = Math.min(adaptive, maximum * 0.025);
  return Math.max(5e-5, capped);
}

export function activeThreshold(rms: Float32Array): number {
  if (rms.length === 0) return 0;
  const maximum = Math.max(...rms);
  if (maximum <= 0) return 0;
  return Math.max(0.010, maximum * 0.040, silenceThreshold(rms) * 1.5);
}

export function activityMask(rms: Float32Array): boolean[] {
  if (rms.length === 0 || Math.max(...rms) <= 0) {
    return Array.from({ length: rms.length }, () => false);
  }
  const threshold = activeThreshold(rms);
  const active = Array.from(rms, (value) => value > threshold);
  if (!active.includes(true)) return active;

  const initiallyActive = active.flatMap((value, index) => value ? [index] : []);
  for (const index of initiallyActive) {
    active[Math.max(0, index - 1)] = true;
  }

  let index = 0;
  while (index < active.length) {
    if (active[index]) {
      index += 1;
      continue;
    }
    const start = index;
    while (index < active.length && !active[index]) index += 1;
    const hasBefore = start > 0 && active[start - 1];
    const hasAfter = index < active.length && active[index];
    if (hasBefore && hasAfter && index - start <= 1) {
      for (let fill = start; fill < index; fill += 1) active[fill] = true;
    }
  }
  return active;
}

export function fallbackPhoneTimeline(rms: Float32Array): Uint8Array {
  const active = activityMask(rms);
  return Uint8Array.from(
    active,
    (value) => value ? FALLBACK_OPEN_PHONE_ID : SILENCE_PHONE_ID,
  );
}
