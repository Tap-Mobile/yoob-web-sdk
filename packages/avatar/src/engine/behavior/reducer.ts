import {
  BEHAVIOR_SCHEMA_VERSION,
  KNOWN_BEHAVIOR_CUES,
  type BehaviorCancellation,
  type BehaviorEnvelope,
  type BehaviorEvent,
  type BehaviorPlan,
  type BehaviorState,
  type FaceOwner,
  type IdentityManifest,
  type KnownBehaviorCue,
} from "./generated/behavior-contract";

export const SPEECH_FACE_YIELD_MS = 80;

export interface BehaviorDispatchResult {
  accepted: boolean;
  plan: BehaviorPlan;
  reason?: "duplicate" | "ignoredOptionalCue" | "staleOffset" | "staleTurn" | "unsupportedCue";
}

const EVENT_TYPES = new Set([
  "sessionStarted",
  "sessionEnded",
  "userSpeechStarted",
  "userSpeechEnded",
  "bargeInCandidateStarted",
  "bargeInCandidateResolved",
  "assistantResponseStarted",
  "assistantAudioStarted",
  "assistantAudioEnded",
  "cue",
]);
const KNOWN_CUES = new Set<string>(KNOWN_BEHAVIOR_CUES);
const RESOLUTIONS = new Set(["falsePositive", "backchannel", "confirmed"]);
const SESSION_END_REASONS = new Set(["normal", "disconnect", "failed"]);

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, allowed: readonly string[], label: string): void {
  const allowedSet = new Set(allowed);
  const unexpected = Object.keys(value).filter((key) => !allowedSet.has(key));
  if (unexpected.length > 0) throw new TypeError(`${label} has unknown field ${unexpected[0]}`);
}

function identifier(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length < 1 || value.length > 160) {
    throw new TypeError(`${label} must be a non-empty identifier of at most 160 characters`);
  }
  return value;
}

function integer(value: unknown, label: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new TypeError(`${label} must be an integer in [${minimum}, ${maximum}]`);
  }
  return value as number;
}

function finiteNumber(value: unknown, label: string, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new TypeError(`${label} must be a finite number in [${minimum}, ${maximum}]`);
  }
  return value;
}

function parseEvent(value: unknown): BehaviorEvent {
  const event = record(value, "event");
  const type = event.type;
  if (typeof type !== "string" || !EVENT_TYPES.has(type)) {
    throw new TypeError(`unsupported behavior event type: ${String(type)}`);
  }
  switch (type) {
    case "sessionStarted":
    case "userSpeechStarted":
    case "userSpeechEnded":
    case "bargeInCandidateStarted":
    case "assistantResponseStarted":
    case "assistantAudioStarted":
    case "assistantAudioEnded":
      exactKeys(event, ["type"], `event ${type}`);
      return { type };
    case "sessionEnded": {
      exactKeys(event, ["type", "reason"], "event sessionEnded");
      const reason = event.reason;
      if (reason !== undefined && (typeof reason !== "string" || !SESSION_END_REASONS.has(reason))) {
        throw new TypeError(`unsupported session end reason: ${String(reason)}`);
      }
      return reason === undefined
        ? { type: "sessionEnded" }
        : { type: "sessionEnded", reason: reason as "normal" | "disconnect" | "failed" };
    }
    case "bargeInCandidateResolved": {
      exactKeys(event, ["type", "resolution", "replacementTurnID"], "event bargeInCandidateResolved");
      if (typeof event.resolution !== "string" || !RESOLUTIONS.has(event.resolution)) {
        throw new TypeError(`unsupported barge-in resolution: ${String(event.resolution)}`);
      }
      const resolution = event.resolution as "falsePositive" | "backchannel" | "confirmed";
      if (resolution === "confirmed") {
        return {
          type,
          resolution,
          replacementTurnID: identifier(event.replacementTurnID, "event.replacementTurnID"),
        };
      }
      if (event.replacementTurnID !== undefined) {
        throw new TypeError("replacementTurnID is valid only for a confirmed barge-in");
      }
      return { type, resolution };
    }
    case "cue":
      exactKeys(event, ["type", "cue", "intensity", "priority", "ttlMs"], "event cue");
      return {
        type,
        cue: identifier(event.cue, "event.cue"),
        intensity: finiteNumber(event.intensity, "event.intensity", 0, 1),
        priority: integer(event.priority, "event.priority", 0, 100),
        ttlMs: integer(event.ttlMs, "event.ttlMs", 1, 10_000),
      };
  }
  throw new TypeError(`unsupported behavior event type: ${String(type)}`);
}

export function parseBehaviorEnvelope(value: unknown): BehaviorEnvelope {
  const envelope = record(value, "behavior envelope");
  exactKeys(
    envelope,
    ["schemaVersion", "eventID", "sessionID", "turnID", "monotonicOffsetMs", "event"],
    "behavior envelope",
  );
  if (envelope.schemaVersion !== BEHAVIOR_SCHEMA_VERSION) {
    throw new TypeError(
      `unsupported behavior schema version ${String(envelope.schemaVersion)}; expected ${BEHAVIOR_SCHEMA_VERSION}`,
    );
  }
  const turnID = envelope.turnID === undefined ? undefined : identifier(envelope.turnID, "turnID");
  return {
    schemaVersion: BEHAVIOR_SCHEMA_VERSION,
    eventID: identifier(envelope.eventID, "eventID"),
    sessionID: identifier(envelope.sessionID, "sessionID"),
    ...(turnID === undefined ? {} : { turnID }),
    monotonicOffsetMs: integer(envelope.monotonicOffsetMs, "monotonicOffsetMs"),
    event: parseEvent(envelope.event),
  };
}

function snapshot(plan: BehaviorPlan): BehaviorPlan {
  return {
    ...plan,
    transition: plan.transition === undefined ? undefined : { ...plan.transition },
    overlays: plan.overlays.map((overlay) => ({ ...overlay })),
    cancellations: plan.cancellations.map((cancellation) => ({ ...cancellation })),
  };
}

export class BehaviorReducer {
  private plan: BehaviorPlan = {
    revision: 0,
    state: "sessionStarting",
    faceOwner: "fallback",
    overlays: [],
    cancellations: [],
  };
  private activeSessionID?: string;
  private lastOffsetMs = -1;
  private readonly seenEventIDs = new Set<string>();
  private readonly tombstonedTurnIDs = new Set<string>();
  private supportedCues = new Set<KnownBehaviorCue>();
  private behaviorAvailable = false;
  private stateBeforeBargeCandidate: BehaviorState = "idle";
  private bargeCandidateTurnID?: string;

  constructor(identity?: IdentityManifest) {
    if (identity !== undefined) this.prepare(identity);
  }

  prepare(identity: IdentityManifest): void {
    const incompatible = identity.requiredBehaviorSchemaVersions.filter(
      (required) => required !== BEHAVIOR_SCHEMA_VERSION,
    );
    if (incompatible.length > 0) {
      throw new TypeError(`unsupported required behavior schema version: ${incompatible.join(", ")}`);
    }
    if (identity.behavior !== undefined && identity.behavior.schemaVersion !== BEHAVIOR_SCHEMA_VERSION) {
      throw new TypeError(`unsupported identity behavior schema version: ${identity.behavior.schemaVersion}`);
    }
    this.behaviorAvailable = identity.behavior !== undefined;
    this.supportedCues = new Set(identity.behavior?.cues ?? []);
    this.plan = { ...this.plan, faceOwner: this.ownerFor(this.plan.state) };
  }

  currentPlan(): BehaviorPlan {
    return snapshot(this.plan);
  }

  dispatch(input: unknown): BehaviorDispatchResult {
    const envelope = parseBehaviorEnvelope(input);
    const startsSession = envelope.event.type === "sessionStarted";
    const startsNewSession = startsSession && envelope.sessionID !== this.activeSessionID;

    if (startsSession) {
      if (this.activeSessionID !== undefined && envelope.sessionID === this.activeSessionID) {
        if (this.seenEventIDs.has(envelope.eventID)) return this.rejected("duplicate");
        throw new TypeError(`session ${envelope.sessionID} cannot be started twice`);
      }
      if (this.activeSessionID !== undefined && this.plan.state !== "ended") {
        throw new TypeError(`session ${this.activeSessionID} is still active`);
      }
    } else if (envelope.sessionID !== this.activeSessionID) {
      throw new TypeError(`event session ${envelope.sessionID} does not match the active session`);
    }

    if (!startsNewSession) {
      if (this.seenEventIDs.has(envelope.eventID)) return this.rejected("duplicate");
      if (envelope.monotonicOffsetMs < this.lastOffsetMs) return this.rejected("staleOffset");
      if (envelope.turnID !== undefined && this.tombstonedTurnIDs.has(envelope.turnID)) {
        return this.rejected("staleTurn");
      }
    }

    // Validate every stateful precondition before consuming the event ID or
    // advancing the monotonic watermark. Callers may correct and resend a
    // malformed-but-parseable event with the same identity.
    this.validateTransition(envelope, startsNewSession);

    if (startsNewSession) {
      this.activeSessionID = envelope.sessionID;
      this.lastOffsetMs = -1;
      this.seenEventIDs.clear();
      this.tombstonedTurnIDs.clear();
      this.bargeCandidateTurnID = undefined;
      this.stateBeforeBargeCandidate = "idle";
    }
    this.seenEventIDs.add(envelope.eventID);
    this.lastOffsetMs = envelope.monotonicOffsetMs;

    const overlays = this.plan.overlays.filter(
      (overlay) => overlay.expiresAtMs > envelope.monotonicOffsetMs,
    );
    const event = envelope.event;
    if (event.type === "cue") {
      if (!KNOWN_CUES.has(event.cue)) return this.rejected("ignoredOptionalCue");
      if (!this.supportedCues.has(event.cue as KnownBehaviorCue)) return this.rejected("unsupportedCue");
      overlays.push({
        cue: event.cue as KnownBehaviorCue,
        intensity: event.intensity,
        priority: event.priority,
        expiresAtMs: envelope.monotonicOffsetMs + event.ttlMs,
        eventID: envelope.eventID,
      });
      return this.commit(envelope, this.plan.state, this.plan.turnID, overlays, []);
    }

    switch (event.type) {
      case "sessionStarted":
        return this.commit(envelope, "idle", undefined, [], []);
      case "sessionEnded": {
        this.bargeCandidateTurnID = undefined;
        const cancellations = this.plan.turnID === undefined
          ? []
          : [{ turnID: this.plan.turnID, reason: "sessionEnded" as const }];
        return this.commit(envelope, "ended", undefined, [], cancellations);
      }
      case "userSpeechStarted":
        return this.commit(envelope, "listening", this.plan.turnID, overlays, []);
      case "userSpeechEnded":
        return this.commit(envelope, "thinking", this.plan.turnID, overlays, []);
      case "bargeInCandidateStarted":
        this.stateBeforeBargeCandidate = this.plan.state;
        this.bargeCandidateTurnID = envelope.turnID;
        return this.commit(envelope, "bargeInCandidate", this.plan.turnID, overlays, []);
      case "bargeInCandidateResolved": {
        this.bargeCandidateTurnID = undefined;
        if (event.resolution !== "confirmed") {
          return this.commit(envelope, this.stateBeforeBargeCandidate, this.plan.turnID, overlays, []);
        }
        const interruptedTurnID = this.requiredTurn(envelope, "confirmed barge-in");
        this.tombstonedTurnIDs.add(interruptedTurnID);
        const cancellation: BehaviorCancellation = {
          turnID: interruptedTurnID,
          reason: "bargeInConfirmed",
        };
        return this.commit(envelope, "listening", event.replacementTurnID, [], [cancellation]);
      }
      case "assistantResponseStarted": {
        const turnID = this.requiredTurn(envelope, "assistantResponseStarted");
        return this.commit(envelope, "assistantBuffering", turnID, overlays, []);
      }
      case "assistantAudioStarted": {
        const turnID = this.requiredCurrentTurn(envelope, "assistantAudioStarted");
        return this.commit(envelope, "assistantSpeaking", turnID, overlays, []);
      }
      case "assistantAudioEnded": {
        const turnID = this.requiredCurrentTurn(envelope, "assistantAudioEnded");
        return this.commit(envelope, "idle", turnID, overlays, []);
      }
    }
  }

  private validateTransition(envelope: BehaviorEnvelope, startsNewSession: boolean): void {
    switch (envelope.event.type) {
      case "sessionStarted":
        if (!startsNewSession) throw new TypeError(`session ${envelope.sessionID} cannot be started twice`);
        return;
      case "bargeInCandidateStarted":
        this.requiredCurrentTurn(envelope, "bargeInCandidateStarted");
        return;
      case "bargeInCandidateResolved": {
        if (this.plan.state !== "bargeInCandidate" || this.bargeCandidateTurnID === undefined) {
          throw new TypeError("bargeInCandidateResolved requires an active barge-in candidate");
        }
        const turnID = this.requiredCurrentTurn(envelope, "bargeInCandidateResolved");
        if (turnID !== this.bargeCandidateTurnID) {
          throw new TypeError("bargeInCandidateResolved does not match the active candidate turn");
        }
        return;
      }
      case "assistantResponseStarted":
        this.requiredTurn(envelope, "assistantResponseStarted");
        return;
      case "assistantAudioStarted":
        this.requiredCurrentTurn(envelope, "assistantAudioStarted");
        return;
      case "assistantAudioEnded":
        this.requiredCurrentTurn(envelope, "assistantAudioEnded");
        return;
      default:
        return;
    }
  }

  private requiredTurn(envelope: BehaviorEnvelope, eventName: string): string {
    if (envelope.turnID === undefined) throw new TypeError(`${eventName} requires turnID`);
    return envelope.turnID;
  }

  private requiredCurrentTurn(envelope: BehaviorEnvelope, eventName: string): string {
    const turnID = this.requiredTurn(envelope, eventName);
    if (this.plan.turnID !== turnID) throw new TypeError(`${eventName} does not match the active turn`);
    return turnID;
  }

  private ownerFor(state: BehaviorState): FaceOwner {
    if (state === "assistantSpeaking") return "speech";
    return this.behaviorAvailable ? "behavior" : "fallback";
  }

  private commit(
    envelope: BehaviorEnvelope,
    state: BehaviorState,
    turnID: string | undefined,
    overlays: BehaviorPlan["overlays"],
    cancellations: BehaviorCancellation[],
  ): BehaviorDispatchResult {
    const from = this.plan.state;
    this.plan = {
      revision: this.plan.revision + 1,
      ...(turnID === undefined ? {} : { turnID }),
      state,
      faceOwner: this.ownerFor(state),
      transition: {
        from,
        to: state,
        startedAtMs: envelope.monotonicOffsetMs,
        durationMs: this.plan.faceOwner === "speech" && state !== "assistantSpeaking"
          ? SPEECH_FACE_YIELD_MS
          : 0,
        reason: envelope.event.type,
      },
      overlays,
      cancellations,
    };
    return { accepted: true, plan: snapshot(this.plan) };
  }

  private rejected(reason: NonNullable<BehaviorDispatchResult["reason"]>): BehaviorDispatchResult {
    return { accepted: false, reason, plan: snapshot(this.plan) };
  }
}
