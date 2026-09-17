export interface StreamingRenderQueueSnapshot {
  pending: number;
  inFlight: number;
  limit: number;
}

/**
 * Bounded producer queue for streaming render windows.
 *
 * Two credits preserve the existing overlap (geometry for N+1 while WebGPU
 * renders N) without allowing an ahead-of-realtime TTS provider to fan out an
 * unbounded set of geometry jobs and renderer/compositor messages.
 */
export class StreamingRenderCreditQueue<T> {
  private readonly pending: T[] = [];
  private inFlight = 0;

  constructor(readonly limit = 2) {
    if (!Number.isSafeInteger(limit) || limit < 1) {
      throw new Error("streaming render credit limit must be a positive integer");
    }
  }

  enqueue(value: T): void {
    this.pending.push(value);
  }

  drain(dispatch: (value: T) => void): number {
    let count = 0;
    while (this.inFlight < this.limit && this.pending.length > 0) {
      const value = this.pending.shift();
      if (value === undefined) break;
      this.inFlight += 1;
      count += 1;
      dispatch(value);
    }
    return count;
  }

  complete(): void {
    if (this.inFlight < 1) {
      throw new Error("streaming render credit completed without an in-flight window");
    }
    this.inFlight -= 1;
  }

  clear(): void {
    this.pending.length = 0;
    this.inFlight = 0;
  }

  snapshot(): StreamingRenderQueueSnapshot {
    return { pending: this.pending.length, inFlight: this.inFlight, limit: this.limit };
  }
}
