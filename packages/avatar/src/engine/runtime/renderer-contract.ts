import { PLANE, RES } from "../assets/serve320-bundle";
import type { RendererSpatialContract } from "./generated/runtime-tier-contract";

// ORT-Web WebGPU mis-packs the first convolution when the channel count is 27.
// The released browser graph pads that convolution to 32 channels with zero
// weights; these five input planes must therefore remain exactly zero.
export const RENDERER_SIGNAL_CHANNELS = 27;
export const RENDERER_INPUT_CHANNELS = 32;
export const RENDERER_OUTPUT_CHANNELS = 3;
export const RENDERER_COMPOSITE_ORDER =
  "host-fill-roi-feather-then-runtime-support-v1" as const;

/** The production/default Serve320 ABI. Omission of the optional contract means this. */
export const FULL320_RENDERER_SPATIAL_CONTRACT = Object.freeze({
  baseWidth: RES,
  baseHeight: RES,
  inputWidth: RES,
  inputHeight: RES,
  originX: 0,
  originY: 0,
  inputChannels: RENDERER_INPUT_CHANNELS,
  signalChannels: RENDERER_SIGNAL_CHANNELS,
  outputChannels: RENDERER_OUTPUT_CHANNELS,
  outputMode: "absoluteRgbV1",
  roiFeatherPixels: 0,
  compositeOrder: RENDERER_COMPOSITE_ORDER,
} satisfies RendererSpatialContract);

/**
 * Experimental native mouth ROI. This is an opt-in ABI description only; no
 * catalog entry or runtime asset selects it.
 */
export const EXPERIMENTAL_224X160_RENDERER_SPATIAL_CONTRACT = Object.freeze({
  ...FULL320_RENDERER_SPATIAL_CONTRACT,
  inputWidth: 224,
  inputHeight: 160,
  originX: 48,
  originY: 96,
  roiFeatherPixels: 6,
} satisfies RendererSpatialContract);

const CONTRACT_KEYS = new Set([
  "baseWidth", "baseHeight", "inputWidth", "inputHeight", "originX", "originY",
  "inputChannels", "signalChannels", "outputChannels", "outputMode",
  "roiFeatherPixels", "compositeOrder",
]);

function integer(value: unknown, name: string, minimum: number): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum) {
    throw new Error(`renderer spatial contract ${name} must be an integer >= ${minimum}`);
  }
  return Number(value);
}

/** Validate the subset implemented by the current 320x320 Web compositor. */
export function validateRendererSpatialContract(
  value: unknown,
): asserts value is RendererSpatialContract {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("renderer spatial contract must be an object");
  }
  const source = value as Record<string, unknown>;
  for (const key of Object.keys(source)) {
    if (!CONTRACT_KEYS.has(key)) throw new Error(`renderer spatial contract has unknown ${key}`);
  }
  const baseWidth = integer(source.baseWidth, "baseWidth", 1);
  const baseHeight = integer(source.baseHeight, "baseHeight", 1);
  const inputWidth = integer(source.inputWidth, "inputWidth", 1);
  const inputHeight = integer(source.inputHeight, "inputHeight", 1);
  const originX = integer(source.originX, "originX", 0);
  const originY = integer(source.originY, "originY", 0);
  const inputChannels = integer(source.inputChannels, "inputChannels", 1);
  const signalChannels = integer(source.signalChannels, "signalChannels", 1);
  const outputChannels = integer(source.outputChannels, "outputChannels", 1);
  const feather = integer(source.roiFeatherPixels, "roiFeatherPixels", 0);
  if (baseWidth !== RES || baseHeight !== RES) {
    throw new Error(`renderer base must be ${RES}x${RES}`);
  }
  if (originX + inputWidth > baseWidth || originY + inputHeight > baseHeight) {
    throw new Error("renderer ROI exceeds the base canvas");
  }
  if (inputChannels !== RENDERER_INPUT_CHANNELS
      || signalChannels !== RENDERER_SIGNAL_CHANNELS
      || outputChannels !== RENDERER_OUTPUT_CHANNELS) {
    throw new Error("renderer channel contract must be 32 input / 27 signal / 3 output");
  }
  if (source.outputMode !== "absoluteRgbV1") {
    throw new Error("renderer outputMode must be absoluteRgbV1");
  }
  if (source.compositeOrder !== RENDERER_COMPOSITE_ORDER) {
    throw new Error(`renderer compositeOrder must be ${RENDERER_COMPOSITE_ORDER}`);
  }
  if (feather * 2 >= Math.min(inputWidth, inputHeight) && feather !== 0) {
    throw new Error("renderer ROI feather consumes the input canvas");
  }
  if (originX === 0 && originY === 0 && inputWidth === RES && inputHeight === RES
      && feather !== 0) {
    throw new Error("full-canvas renderer must not add an ROI edge feather");
  }
}

export function resolveRendererSpatialContract(
  value?: RendererSpatialContract,
): RendererSpatialContract {
  if (value === undefined) return FULL320_RENDERER_SPATIAL_CONTRACT;
  validateRendererSpatialContract(value);
  return value;
}

export function isRendererSpatialContract(value: unknown): value is RendererSpatialContract {
  try {
    validateRendererSpatialContract(value);
    return true;
  } catch {
    return false;
  }
}

export function isFull320RendererContract(contract: RendererSpatialContract): boolean {
  return contract.baseWidth === RES && contract.baseHeight === RES
    && contract.inputWidth === RES && contract.inputHeight === RES
    && contract.originX === 0 && contract.originY === 0
    && contract.roiFeatherPixels === 0;
}

export function rendererPlane(contract: RendererSpatialContract): number {
  return contract.inputWidth * contract.inputHeight;
}

export function rendererInputLength(contract: RendererSpatialContract): number {
  return contract.inputChannels * rendererPlane(contract);
}

export function rendererOutputLength(contract: RendererSpatialContract): number {
  return contract.outputChannels * rendererPlane(contract);
}

export function rendererInputShape(contract: RendererSpatialContract): number[] {
  return [1, contract.inputChannels, contract.inputHeight, contract.inputWidth];
}

export function rendererOutputShape(contract: RendererSpatialContract): number[] {
  return [1, contract.outputChannels, contract.inputHeight, contract.inputWidth];
}

export const RENDERER_INPUT_LENGTH = RENDERER_INPUT_CHANNELS * PLANE;

export function clearRendererConditioning(
  input: Float32Array,
  contract: RendererSpatialContract = FULL320_RENDERER_SPATIAL_CONTRACT,
): void {
  const plane = rendererPlane(contract);
  const expected = rendererInputLength(contract);
  if (input.length !== expected) throw new Error(`bad renderer input ${input.length}`);
  // Channels 0..6 are host/reference/mask and are overwritten by the caller.
  // Clear heatmaps 7..26 and the WebGPU padding planes 27..31.
  input.fill(0, 7 * plane);
}

/**
 * The normal renderer path overwrites all 20 heatmap planes on every neural
 * frame. Only the five WebGPU padding planes therefore need an explicit clear.
 */
export function clearRendererPadding(
  input: Float32Array,
  contract: RendererSpatialContract = FULL320_RENDERER_SPATIAL_CONTRACT,
): void {
  const plane = rendererPlane(contract);
  const expected = rendererInputLength(contract);
  if (input.length !== expected) throw new Error(`bad renderer input ${input.length}`);
  input.fill(0, contract.signalChannels * plane);
}
