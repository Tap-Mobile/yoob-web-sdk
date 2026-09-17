export interface MicrophoneOption {
  deviceId: string;
  label: string;
}

export interface OpenMicrophoneResult {
  stream: MediaStream;
  usedFallback: boolean;
}

export function microphoneConstraints(inputDeviceId: string | null): MediaStreamConstraints {
  const audio: MediaTrackConstraints = {
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };
  if (inputDeviceId) audio.deviceId = { exact: inputDeviceId };
  return { audio, video: false };
}

function unavailableDevice(error: unknown): boolean {
  const name = typeof error === "object" && error && "name" in error
    ? String(error.name)
    : "";
  return name === "NotFoundError" || name === "OverconstrainedError";
}

/**
 * Open the saved microphone, falling back only when that physical device has
 * disappeared. Permission and hardware-in-use failures must remain visible to
 * the user instead of triggering a second, confusing permission request.
 */
export async function openPreferredMicrophone(
  mediaDevices: Pick<MediaDevices, "getUserMedia">,
  inputDeviceId: string | null,
): Promise<OpenMicrophoneResult> {
  try {
    return {
      stream: await mediaDevices.getUserMedia(microphoneConstraints(inputDeviceId)),
      usedFallback: false,
    };
  } catch (error) {
    if (!inputDeviceId || !unavailableDevice(error)) throw error;
    return {
      stream: await mediaDevices.getUserMedia(microphoneConstraints(null)),
      usedFallback: true,
    };
  }
}

export async function enumerateMicrophones(
  mediaDevices: Pick<MediaDevices, "enumerateDevices">,
): Promise<MicrophoneOption[]> {
  const devices = await mediaDevices.enumerateDevices();
  const seen = new Set<string>();
  const microphones: MicrophoneOption[] = [];
  for (const device of devices) {
    if (device.kind !== "audioinput" || !device.deviceId || device.deviceId === "default"
        || seen.has(device.deviceId)) continue;
    seen.add(device.deviceId);
    const label = device.label.trim().slice(0, 160);
    microphones.push({
      deviceId: device.deviceId,
      label: label || `Microphone ${microphones.length + 1}`,
    });
  }
  return microphones;
}

export function normalizedMicrophoneLevel(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sumSquares = 0;
  for (const sample of samples) sumSquares += sample * sample;
  // Speech RMS is commonly well below full scale; expand it for a useful UI
  // while keeping the value bounded for progressbar semantics.
  return Math.min(1, Math.sqrt(sumSquares / samples.length) * 4);
}

export class MicrophonePreview {
  private stream?: MediaStream;
  private context?: AudioContext;
  private source?: MediaStreamAudioSourceNode;
  private analyser?: AnalyserNode;
  private frame?: number;

  async start(
    mediaDevices: Pick<MediaDevices, "getUserMedia">,
    inputDeviceId: string | null,
    onLevel: (level: number) => void,
  ): Promise<OpenMicrophoneResult> {
    this.stop();
    const opened = await openPreferredMicrophone(mediaDevices, inputDeviceId);
    try {
      const context = new AudioContext({ latencyHint: "interactive" });
      const source = context.createMediaStreamSource(opened.stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.72;
      source.connect(analyser);
      await context.resume();
      this.stream = opened.stream;
      this.context = context;
      this.source = source;
      this.analyser = analyser;
      const samples = new Float32Array(analyser.fftSize);
      const update = () => {
        if (this.analyser !== analyser) return;
        analyser.getFloatTimeDomainData(samples);
        onLevel(normalizedMicrophoneLevel(samples));
        this.frame = requestAnimationFrame(update);
      };
      update();
      return opened;
    } catch (error) {
      opened.stream.getTracks().forEach((track) => track.stop());
      throw error;
    }
  }

  stop(): void {
    if (this.frame !== undefined) cancelAnimationFrame(this.frame);
    this.frame = undefined;
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    void this.context?.close();
    this.stream = undefined;
    this.context = undefined;
    this.source = undefined;
    this.analyser = undefined;
  }
}
