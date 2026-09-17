import { PLANE, RES } from "../assets/serve320-bundle";
import type { Point } from "../math/geometry";
import type { RendererSpatialContract } from "./generated/runtime-tier-contract";
import { rendererOutputToBgrInto, writeHeatmapsInto } from "./frame-scratch";
import {
  FULL320_RENDERER_SPATIAL_CONTRACT,
  clearRendererConditioning,
  clearRendererPadding,
  isFull320RendererContract,
  rendererInputLength,
  rendererOutputLength,
  rendererPlane,
  resolveRendererSpatialContract,
} from "./renderer-contract";

export interface RendererInputBuilderScratch {
  heatmapGaussianX: Float32Array;
  heatmapGaussianY: Float32Array;
  rendererPoints: Point[];
}

export function shiftRendererLandmarksInto(
  points: Point[],
  contract: RendererSpatialContract,
  destination: Point[],
): Point[] {
  if (destination.length < points.length) {
    throw new Error(`renderer landmark scratch mismatch ${destination.length}/${points.length}`);
  }
  for (let index = 0; index < points.length; index += 1) {
    destination[index][0] = points[index][0] - contract.originX;
    destination[index][1] = points[index][1] - contract.originY;
  }
  return destination;
}

/**
 * Pack the exact Serve renderer channel ABI. Inputs remain full-320 after host
 * warp and reference alignment; an opt-in ROI contract crops only at this last
 * stage and shifts the landmark coordinate system before heatmap generation.
 */
export function buildRendererInput(
  host: Uint8Array,
  alignedReference: Float32Array,
  points: Point[],
  hole: Float32Array,
  input: Float32Array,
  scratch: RendererInputBuilderScratch,
  spatial: RendererSpatialContract = FULL320_RENDERER_SPATIAL_CONTRACT,
): void {
  const contract = spatial;
  const basePlane = contract.baseWidth * contract.baseHeight;
  const plane = rendererPlane(contract);
  if (host.length !== 3 * basePlane
      || alignedReference.length !== 3 * basePlane
      || hole.length !== basePlane) {
    throw new Error("renderer full-canvas conditioning shape mismatch");
  }
  if (input.length !== rendererInputLength(contract)) {
    throw new Error(`bad renderer input ${input.length}`);
  }
  const fullCanvas = isFull320RendererContract(contract);
  if (fullCanvas) {
    // Preserve the incumbent hot loop exactly when the optional contract is
    // omitted. This is both byte-identical and avoids ROI address arithmetic.
    for (let pixel = 0; pixel < PLANE; pixel += 1) {
      const keep = 1 - hole[pixel];
      input[pixel] = host[pixel * 3 + 2] / 255 * keep;
      input[PLANE + pixel] = host[pixel * 3 + 1] / 255 * keep;
      input[2 * PLANE + pixel] = host[pixel * 3] / 255 * keep;
      input[3 * PLANE + pixel] = alignedReference[pixel];
      input[4 * PLANE + pixel] = alignedReference[PLANE + pixel];
      input[5 * PLANE + pixel] = alignedReference[2 * PLANE + pixel];
      input[6 * PLANE + pixel] = hole[pixel];
    }
  } else {
    for (let y = 0; y < contract.inputHeight; y += 1) {
      const sourceRow = (contract.originY + y) * contract.baseWidth + contract.originX;
      const destinationRow = y * contract.inputWidth;
      for (let x = 0; x < contract.inputWidth; x += 1) {
        const sourcePixel = sourceRow + x;
        const pixel = destinationRow + x;
        const keep = 1 - hole[sourcePixel];
        input[pixel] = host[sourcePixel * 3 + 2] / 255 * keep;
        input[plane + pixel] = host[sourcePixel * 3 + 1] / 255 * keep;
        input[2 * plane + pixel] = host[sourcePixel * 3] / 255 * keep;
        input[3 * plane + pixel] = alignedReference[sourcePixel];
        input[4 * plane + pixel] = alignedReference[basePlane + sourcePixel];
        input[5 * plane + pixel] = alignedReference[2 * basePlane + sourcePixel];
        input[6 * plane + pixel] = hole[sourcePixel];
      }
    }
  }
  // All 20 heatmap channels are rewritten below for the 20-point contract, so
  // only the exact-zero WebGPU padding planes need clearing.
  if (points.length === 20) clearRendererPadding(input, contract);
  else clearRendererConditioning(input, contract);
  const rendererPoints = fullCanvas
    ? points
    : shiftRendererLandmarksInto(points, contract, scratch.rendererPoints);
  writeHeatmapsInto(
    rendererPoints,
    input,
    scratch.heatmapGaussianX,
    scratch.heatmapGaussianY,
    7,
    4,
    contract.inputWidth,
    contract.inputHeight,
    points.length,
  );
}

/**
 * Full-320 multiplier for the existing dynamic compositor support. Outside the
 * ROI it is zero; the ROI's rectangular edge reaches one after the configured
 * feather distance. The full renderer returns undefined so its hot path does
 * not gain an extra multiply or allocation.
 */
export function rendererSupportMultiplier(
  spatial: RendererSpatialContract,
): Float32Array | undefined {
  const contract = resolveRendererSpatialContract(spatial);
  if (isFull320RendererContract(contract)) return undefined;
  const output = new Float32Array(contract.baseWidth * contract.baseHeight);
  const feather = contract.roiFeatherPixels;
  const ramp = (coordinate: number, size: number): number => {
    if (feather === 0) return 1;
    return Math.min(1, Math.min(coordinate, size - 1 - coordinate) / feather);
  };
  for (let y = 0; y < contract.inputHeight; y += 1) {
    const yWeight = ramp(y, contract.inputHeight);
    const row = (contract.originY + y) * contract.baseWidth + contract.originX;
    for (let x = 0; x < contract.inputWidth; x += 1) {
      output[row + x] = yWeight * ramp(x, contract.inputWidth);
    }
  }
  return output;
}

/**
 * Convert the renderer's planar RGB result to the existing full-320 BGR host
 * canvas. ROI feathering is deliberately not baked into these pixels: the
 * compositor multiplies its generated support by rendererSupportMultiplier(),
 * preserving the ordered host-fill -> ROI feather -> runtime-support contract.
 */
export function expandRendererOutputToFullBgrInto(
  output: Float32Array,
  rendererHost: Uint8Array,
  destination: Uint8Array,
  spatial: RendererSpatialContract,
): void {
  const contract = spatial;
  if (output.length !== rendererOutputLength(contract)) {
    throw new Error(`bad renderer output ${output.length}`);
  }
  if (rendererHost.length !== 3 * PLANE || destination.length !== 3 * PLANE) {
    throw new Error("renderer host-fill canvas must be full320 BGR");
  }
  if (isFull320RendererContract(contract)) {
    rendererOutputToBgrInto(output, destination, RES, RES);
    return;
  }
  destination.set(rendererHost);
  const plane = rendererPlane(contract);
  for (let y = 0; y < contract.inputHeight; y += 1) {
    const destinationRow = (contract.originY + y) * contract.baseWidth + contract.originX;
    const sourceRow = y * contract.inputWidth;
    for (let x = 0; x < contract.inputWidth; x += 1) {
      const sourcePixel = sourceRow + x;
      const destinationPixel = destinationRow + x;
      const red = Math.min(1, Math.max(0, (output[sourcePixel] + 1) / 2));
      const green = Math.min(1, Math.max(0, (output[plane + sourcePixel] + 1) / 2));
      const blue = Math.min(1, Math.max(0, (output[2 * plane + sourcePixel] + 1) / 2));
      destination[destinationPixel * 3] = Math.min(255, Math.floor(blue * 255 + 0.5));
      destination[destinationPixel * 3 + 1] = Math.min(255, Math.floor(green * 255 + 0.5));
      destination[destinationPixel * 3 + 2] = Math.min(255, Math.floor(red * 255 + 0.5));
    }
  }
}
