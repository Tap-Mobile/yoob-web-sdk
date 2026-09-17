export const MICROPHONE_PACKET_STALL_MS = 1_600;
export const MICROPHONE_WATCHDOG_INTERVAL_MS = 500;
export const MICROPHONE_SOFT_RECOVERY_MS = 300;
export const MICROPHONE_REOPEN_RECOVERY_MS = 800;

/**
 * The mic worklet emits a packet every 20 ms even for silence. If neither the
 * initial enable nor the last packet is recent, the capture graph is stalled —
 * this is not speech detection and therefore does not mistake silence for a
 * dead microphone.
 */
export function microphonePcmStalled(
  enabledAtMs: number,
  lastPacketAtMs: number,
  nowMs: number,
  stallMs = MICROPHONE_PACKET_STALL_MS,
): boolean {
  if (enabledAtMs <= 0 || nowMs < enabledAtMs) return false;
  const lastActivity = Math.max(enabledAtMs, lastPacketAtMs);
  return nowMs - lastActivity >= stallMs;
}
