export async function fetchArrayBuffer(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length") || 0);
  if (!response.body) {
    const value = await response.arrayBuffer();
    onProgress?.(value.byteLength, value.byteLength);
    return value;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const joined = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return joined.buffer;
}

export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function assertLength(name: string, actual: number, expected: number): void {
  if (actual !== expected) throw new Error(`${name}: ${actual} bytes != ${expected}`);
}

export function isLittleEndian(): boolean {
  const bytes = new Uint8Array([1, 0]);
  return new Uint16Array(bytes.buffer)[0] === 1;
}
