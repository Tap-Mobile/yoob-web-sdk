import { roundToEven } from "../math/rounding";
import type { Point } from "../math/geometry";

export function dilateDownBinary(
  source: Float32Array,
  width: number,
  height: number,
  depth: number,
  destination?: Float32Array,
  window?: RectWindow,
) {
  if (depth <= 0) {
    if (destination) {
      destination.set(source);
      return destination;
    }
    return source.slice();
  }
  const output = destination ?? new Float32Array(source.length);
  if (destination) output.fill(0);
  // Columns outside the source's support can never receive a downward smear,
  // so callers pass the support bbox as `window`. The scan still runs from the
  // top of each column (a source pixel above the window dilates into it);
  // only the writes are windowed, which keeps values identical.
  const x0 = window?.x0 ?? 0, x1 = window?.x1 ?? width - 1;
  const y0 = window?.y0 ?? 0, y1 = window?.y1 ?? height - 1;
  for (let x = x0; x <= x1; x += 1) {
    let last = -1_000_000;
    for (let y = 0; y < height; y += 1) {
      if (source[y * width + x] > 0) last = y;
      if (y >= y0 && y <= y1 && y - last <= depth) output[y * width + x] = 1;
    }
  }
  return output;
}

export interface RectWindow {
  x0: number; y0: number; x1: number; y1: number;
}

/**
 * Erosion by the 5x5 ellipse stencil (all taps except the four corners).
 *
 * The stencil is decomposed exactly: a tap set {ky=0,±1: kx∈[-2,2]} plus
 * {ky=±2: kx∈[-1,1]} is min(h3[y-2], h5[y-1], h5[y], h5[y+1], h3[y+2]) where
 * h3/h5 are horizontal sliding minima of radius 1/2, and h5 is itself the
 * radius-1 minimum of h3. Min is order-independent and exact, so this is
 * bitwise identical to the 24-tap nested loop with the same skip-OOB edges,
 * at roughly one third of the comparisons.
 *
 * `window` restricts the computed output region; reads still see the real
 * source array, so windowed results equal the full computation inside the
 * window. Cells outside the window are left untouched (caller zeroes).
 */
export function erodeEllipse5x5(
  source: Float32Array,
  width: number,
  height: number,
  destination?: Float32Array,
  scratchA?: Float32Array,
  scratchB?: Float32Array,
  window?: RectWindow,
) {
  const output = destination ?? new Float32Array(source.length);
  const h3 = scratchA ?? new Float32Array(source.length);
  const h5 = scratchB ?? new Float32Array(source.length);
  const x0 = window?.x0 ?? 0, x1 = window?.x1 ?? width - 1;
  const y0 = window?.y0 ?? 0, y1 = window?.y1 ?? height - 1;
  const rowY0 = Math.max(0, y0 - 2), rowY1 = Math.min(height - 1, y1 + 2);
  // h5[x] consumes h3[x±1], so h3 spans one extra cell on each side.
  const hx0 = Math.max(0, x0 - 1), hx1 = Math.min(width - 1, x1 + 1);
  for (let y = rowY0; y <= rowY1; y += 1) {
    const row = y * width;
    for (let x = hx0; x <= hx1; x += 1) {
      let minimum = source[row + x];
      if (x > 0) minimum = Math.min(minimum, source[row + x - 1]);
      if (x < width - 1) minimum = Math.min(minimum, source[row + x + 1]);
      h3[row + x] = minimum;
    }
    // h5 reads h3 at x+1, so it runs strictly after the h3 row is complete.
    for (let x = x0; x <= x1; x += 1) {
      let minimum = h3[row + x];
      if (x > 0) minimum = Math.min(minimum, h3[row + x - 1]);
      if (x < width - 1) minimum = Math.min(minimum, h3[row + x + 1]);
      h5[row + x] = minimum;
    }
  }
  for (let y = y0; y <= y1; y += 1) {
    const row = y * width;
    for (let x = x0; x <= x1; x += 1) {
      let minimum = Number.POSITIVE_INFINITY;
      if (y > 0) minimum = h5[row - width + x];
      minimum = Math.min(minimum, h5[row + x]);
      if (y < height - 1) minimum = Math.min(minimum, h5[row + width + x]);
      if (y > 1) minimum = Math.min(minimum, h3[row - 2 * width + x]);
      if (y < height - 2) minimum = Math.min(minimum, h3[row + 2 * width + x]);
      output[row + x] = Number.isFinite(minimum) ? minimum : 0;
    }
  }
  return output;
}

function reflect101(value: number, size: number): number {
  if (value < 0) return -value;
  if (value >= size) return 2 * (size - 1) - value;
  return value;
}

export function gaussianBlur(
  source: Float32Array,
  width: number,
  height: number,
  sigma: number,
  destination?: Float32Array,
  temporary?: Float32Array,
  kernelScratch?: Float32Array,
  window?: RectWindow,
) {
  const kernelSize = roundToEven(sigma * 8 + 1) | 1;
  const center = (kernelSize - 1) * 0.5;
  const kernel = kernelScratch && kernelScratch.length >= kernelSize
    ? kernelScratch.subarray(0, kernelSize)
    : new Float32Array(kernelSize);
  let total = 0;
  for (let i = 0; i < kernelSize; i += 1) {
    const position = (i - center) / sigma;
    kernel[i] = Math.exp(-0.5 * position * position);
    total += kernel[i];
  }
  for (let i = 0; i < kernelSize; i += 1) kernel[i] /= total;
  const radius = Math.floor(kernelSize / 2);
  const temp = temporary ?? new Float32Array(source.length);
  const output = destination ?? new Float32Array(source.length);
  // Windowed callers get only the output region they ask for; the horizontal
  // pass then covers every row the vertical pass reads — including rows that
  // only appear through reflect101 at the plane borders. Tap order and
  // reflect101 edges are unchanged, so values are bitwise identical to the
  // full-plane computation inside the window.
  const x0 = window?.x0 ?? 0, x1 = window?.x1 ?? width - 1;
  const y0 = window?.y0 ?? 0, y1 = window?.y1 ?? height - 1;
  const tempY0 = Math.max(0, Math.min(y0 - radius, 2 * (height - 1) - (y1 + radius)));
  const tempY1 = Math.min(height - 1, Math.max(y1 + radius, radius - y0));
  for (let y = tempY0; y <= tempY1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      let value = 0;
      for (let k = 0; k < kernelSize; k += 1) {
        value += source[y * width + reflect101(x + k - radius, width)] * kernel[k];
      }
      temp[y * width + x] = value;
    }
  }
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      let value = 0;
      for (let k = 0; k < kernelSize; k += 1) {
        value += temp[reflect101(y + k - radius, height) * width + x] * kernel[k];
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

// cv2 INTER_LINEAR geometry used by the native canonical compositor:
// src = (dst + 0.5) * scale - 0.5, with edge taps replicated.
export function resizeBilinearFloat(
  source: Float32Array, sourceWidth: number, sourceHeight: number,
  targetWidth: number, targetHeight: number,
  destination?: Float32Array,
): Float32Array {
  const output = destination ?? new Float32Array(targetWidth * targetHeight);
  if (destination && destination.length < targetWidth * targetHeight) {
    throw new Error("resizeBilinearFloat destination too small");
  }
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = (y + 0.5) * scaleY - 0.5;
    let y0 = Math.floor(sourceY);
    let weightY = sourceY - y0;
    if (y0 < 0) { y0 = 0; weightY = 0; }
    if (y0 >= sourceHeight - 1) { y0 = sourceHeight - 1; weightY = 0; }
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = (x + 0.5) * scaleX - 0.5;
      let x0 = Math.floor(sourceX);
      let weightX = sourceX - x0;
      if (x0 < 0) { x0 = 0; weightX = 0; }
      if (x0 >= sourceWidth - 1) { x0 = sourceWidth - 1; weightX = 0; }
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const top = source[y0 * sourceWidth + x0] * (1 - weightX)
        + source[y0 * sourceWidth + x1] * weightX;
      const bottom = source[y1 * sourceWidth + x0] * (1 - weightX)
        + source[y1 * sourceWidth + x1] * weightX;
      output[y * targetWidth + x] = top * (1 - weightY) + bottom * weightY;
    }
  }
  return destination ? destination.subarray(0, targetWidth * targetHeight) : output;
}

export function resizeBilinearU8(
  source: Uint8Array, sourceWidth: number, sourceHeight: number,
  targetWidth: number, targetHeight: number,
): Uint8Array {
  const asFloat = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) asFloat[index] = source[index];
  const resized = resizeBilinearFloat(
    asFloat, sourceWidth, sourceHeight, targetWidth, targetHeight,
  );
  const output = new Uint8Array(resized.length);
  for (let index = 0; index < resized.length; index += 1) {
    output[index] = Math.min(255, Math.max(0, Math.floor(resized[index] + 0.5)));
  }
  return output;
}

/**
 * One-dimensional sliding-window extremum with the same clamped edge
 * semantics as the original nested loop. Interior pixels use block
 * prefix/suffix extrema, making the cost independent of the window radius.
 */
function lineExtrema(
  source: Float32Array,
  base: number,
  stride: number,
  length: number,
  radius: number,
  dilate: boolean,
  destination: Float32Array,
  scratch: Float32Array,
): void {
  const size = 2 * radius + 1;
  const initial = dilate ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  const forward = scratch.subarray(0, length);
  const backward = scratch.subarray(length, 2 * length);
  const prefix = scratch.subarray(2 * length, 3 * length);
  const suffix = scratch.subarray(3 * length, 4 * length);
  for (let blockStart = 0; blockStart < length; blockStart += size) {
    const blockEnd = Math.min(length - 1, blockStart + size - 1);
    let value = initial;
    for (let index = blockStart; index <= blockEnd; index += 1) {
      const sample = source[base + index * stride];
      value = dilate ? Math.max(value, sample) : Math.min(value, sample);
      forward[index] = value;
    }
    value = initial;
    for (let index = blockEnd; index >= blockStart; index -= 1) {
      const sample = source[base + index * stride];
      value = dilate ? Math.max(value, sample) : Math.min(value, sample);
      backward[index] = value;
    }
  }
  let value = initial;
  for (let index = 0; index < length; index += 1) {
    const sample = source[base + index * stride];
    value = dilate ? Math.max(value, sample) : Math.min(value, sample);
    prefix[index] = value;
  }
  value = initial;
  for (let index = length - 1; index >= 0; index -= 1) {
    const sample = source[base + index * stride];
    value = dilate ? Math.max(value, sample) : Math.min(value, sample);
    suffix[index] = value;
  }
  for (let index = 0; index < length; index += 1) {
    const low = index - radius;
    const high = index + radius;
    if (low < 0) value = prefix[Math.min(high, length - 1)];
    else if (high >= length) value = suffix[low];
    else if (dilate) value = Math.max(backward[low], forward[high]);
    else value = Math.min(backward[low], forward[high]);
    destination[base + index * stride] = value;
  }
}

function rectExtrema(
  source: Float32Array,
  width: number,
  height: number,
  size: number,
  dilate: boolean,
  destination?: Float32Array,
  temporary?: Float32Array,
  window?: RectWindow,
) {
  const radius = Math.floor(size / 2);
  const temp = temporary ?? new Float32Array(source.length);
  const output = destination ?? new Float32Array(source.length);
  const lineLength = Math.max(width, height);
  const scratch = new Float32Array(4 * lineLength);
  if (!window) {
    for (let y = 0; y < height; y += 1) {
      lineExtrema(source, y * width, 1, width, radius, dilate, temp, scratch);
    }
    for (let x = 0; x < width; x += 1) {
      lineExtrema(temp, x, width, height, radius, dilate, output, scratch);
    }
    return output;
  }
  // Windowed path: identical clamped-window extrema, computed only where the
  // caller consumes them. The horizontal pass covers the rows the vertical
  // pass reads; the deque slides only across the needed range but sees the
  // real source values, so results equal the full-line computation.
  const x0 = window.x0, x1 = window.x1, y0 = window.y0, y1 = window.y1;
  // The window covers the true support; everything outside it is zero.
  if (destination) output.fill(0);
  for (let y = Math.max(0, y0 - radius); y <= Math.min(height - 1, y1 + radius); y += 1) {
    lineExtremaRange(source, y * width, 1, width, x0, x1, radius, dilate, temp);
  }
  for (let x = x0; x <= x1; x += 1) {
    lineExtremaRange(temp, x, width, height, y0, y1, radius, dilate, output);
  }
  return output;
}

/**
 * Sliding-window extremum restricted to output indices [outStart, outEnd]
 * using a monotonic deque. Values are the same clamped-edge extrema as
 * `lineExtrema` — the deque sees every source sample the full-line window
 * would see; only the emits are ranged.
 */
let dequeScratch = new Int32Array(0);

function lineExtremaRange(
  source: Float32Array,
  base: number,
  stride: number,
  length: number,
  outStart: number,
  outEnd: number,
  radius: number,
  dilate: boolean,
  destination: Float32Array,
): void {
  if (dequeScratch.length < length + 1) dequeScratch = new Int32Array(length + 1);
  const deque = dequeScratch;
  let head = 0, tail = 0;
  for (let index = outStart; index <= outEnd; index += 1) {
    const low = Math.max(0, index - radius);
    const high = Math.min(length - 1, index + radius);
    if (index === outStart) {
      for (let j = low; j <= high; j += 1) {
        while (tail > head
            && (dilate
              ? source[base + j * stride] >= source[base + deque[tail - 1] * stride]
              : source[base + j * stride] <= source[base + deque[tail - 1] * stride])) {
          tail -= 1;
        }
        deque[tail] = j;
        tail += 1;
      }
    } else {
      while (tail > head
          && (dilate
            ? source[base + high * stride] >= source[base + deque[tail - 1] * stride]
            : source[base + high * stride] <= source[base + deque[tail - 1] * stride])) {
        tail -= 1;
      }
      deque[tail] = high;
      tail += 1;
      if (deque[head] < low) head += 1;
    }
    destination[base + index * stride] = source[base + deque[head] * stride];
  }
}

export function dilateRect(
  source: Float32Array,
  width: number,
  height: number,
  size: number,
  destination?: Float32Array,
  temporary?: Float32Array,
  window?: RectWindow,
) {
  return rectExtrema(source, width, height, size, true, destination, temporary, window);
}

export function morphCloseRect(
  source: Float32Array,
  width: number,
  height: number,
  size: number,
  destination?: Float32Array,
  temporary?: Float32Array,
  intermediate?: Float32Array,
  window?: RectWindow,
) {
  const mid = intermediate ?? new Float32Array(source.length);
  const temp = temporary ?? new Float32Array(source.length);
  if (!window) {
    const dilated = rectExtrema(source, width, height, size, true, mid, temp);
    return rectExtrema(dilated, width, height, size, false, destination, temp);
  }
  // The closing is consumed only inside `window`; the dilation must cover the
  // window expanded by the radius (the erosion reads it), nothing more.
  const radius = Math.floor(size / 2);
  const dilated = rectExtrema(source, width, height, size, true, mid, temp, {
    x0: Math.max(0, window.x0 - radius),
    y0: Math.max(0, window.y0 - radius),
    x1: Math.min(width - 1, window.x1 + radius),
    y1: Math.min(height - 1, window.y1 + radius),
  });
  return rectExtrema(dilated, width, height, size, false, destination, temp, window);
}

export function erodeRectBinary(
  source: Float32Array,
  width: number,
  height: number,
  size: number,
  destination?: Float32Array,
  integralScratch?: Int32Array,
  window?: RectWindow,
) {
  const radius = Math.floor(size / 2);
  const stride = width + 1;
  const needed = (width + 1) * (height + 1);
  const integral = integralScratch && integralScratch.length >= needed
    ? integralScratch
    : new Int32Array(needed);
  const output = destination ?? new Float32Array(source.length);
  if (destination) output.fill(0);
  // Only the window's output cells are computed. The integral covers the
  // window expanded by the radius — every clamped rect corner lands inside it
  // — and integer sums make the result identical to the full-plane pass.
  const x0 = window?.x0 ?? 0, y0 = window?.y0 ?? 0;
  const x1 = window?.x1 ?? width - 1, y1 = window?.y1 ?? height - 1;
  const ix0 = Math.max(0, x0 - radius), iy0 = Math.max(0, y0 - radius);
  const ix1 = Math.min(width - 1, x1 + radius), iy1 = Math.min(height - 1, y1 + radius);
  // Zero the boundary row and column the recurrence reads.
  for (let x = ix0; x <= ix1 + 1; x += 1) integral[iy0 * stride + x] = 0;
  for (let y = iy0; y <= iy1; y += 1) {
    integral[(y + 1) * stride + ix0] = 0;
    let rowSum = 0;
    for (let x = ix0; x <= ix1; x += 1) {
      if (source[y * width + x] > 0) rowSum += 1;
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
    }
  }
  for (let y = y0; y <= y1; y += 1) {
    const ry0 = Math.max(0, y - radius), ry1 = Math.min(height - 1, y + radius);
    for (let x = x0; x <= x1; x += 1) {
      const rx0 = Math.max(0, x - radius), rx1 = Math.min(width - 1, x + radius);
      const sum = integral[(ry1 + 1) * stride + rx1 + 1]
        - integral[ry0 * stride + rx1 + 1]
        - integral[(ry1 + 1) * stride + rx0]
        + integral[ry0 * stride + rx0];
      if (sum === (rx1 - rx0 + 1) * (ry1 - ry0 + 1)) output[y * width + x] = 1;
    }
  }
  return output;
}

export function convexHull(points: Point[]): Point[] {
  const sorted = points.map(([x, y]) => [Math.trunc(x), Math.trunc(y)] as Point)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length <= 2) return sorted;
  const cross = (origin: Point, a: Point, b: Point) =>
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);
  const lower: Point[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower.at(-1)!, point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Point[] = [];
  for (const point of [...sorted].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper.at(-1)!, point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

export function fillConvexPoly(
  hull: Point[],
  width: number,
  height: number,
  destination?: Float32Array,
) {
  const output = destination ?? new Float32Array(width * height);
  if (destination) output.fill(0);
  if (hull.length < 3) return output;
  const minY = Math.max(0, Math.min(...hull.map((point) => point[1])));
  const maxY = Math.min(height - 1, Math.max(...hull.map((point) => point[1])));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let inside = true;
      for (let i = 0; i < hull.length; i += 1) {
        const a = hull[i], b = hull[(i + 1) % hull.length];
        const value = (b[0] - a[0]) * (y + 0.5 - a[1]) - (b[1] - a[1]) * (x + 0.5 - a[0]);
        if (value < 0) { inside = false; break; }
      }
      if (inside) output[y * width + x] = 1;
    }
  }
  return output;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
