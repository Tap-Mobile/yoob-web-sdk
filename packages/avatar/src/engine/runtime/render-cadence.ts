// Quality and Balanced render one neural mouth for every 25-fps host frame.
// Fast may request stride 2 only through its immutable, quality-gated runtime
// descriptor. No query flag or device heuristic can change cadence in-place.
export type NeuralMouthRenderStride = 1 | 2;
export const NEURAL_MOUTH_RENDER_STRIDE: NeuralMouthRenderStride = 1;

export function neuralMouthRenderStride(value: unknown): NeuralMouthRenderStride {
  if (value === undefined || value === 1) return 1;
  if (value === 2) return 2;
  throw new Error(`unsupported neural mouth stride ${String(value)}`);
}

export function shouldRunNeuralMouth(
  frameIndex: number,
  stride: NeuralMouthRenderStride = NEURAL_MOUTH_RENDER_STRIDE,
): boolean {
  return frameIndex % stride === 0;
}

export function expectedNeuralMouthFrames(
  frameOffset: number,
  frameCount: number,
  stride: NeuralMouthRenderStride = NEURAL_MOUTH_RENDER_STRIDE,
): number {
  let count = 0;
  for (let local = 0; local < frameCount; local += 1) {
    if (shouldRunNeuralMouth(frameOffset + local, stride)) count += 1;
  }
  return count;
}
