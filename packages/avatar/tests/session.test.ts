import { test } from "node:test";
import assert from "node:assert/strict";
import { SessionMonitor, type SessionMonitorOptions } from "../src/session";
import { YoobError } from "../src/cdn";

type Reply = { status: number; body?: unknown } | Error;

/** A heartbeat endpoint that answers from a script, and a clock that only moves when the test says so. */
function harness(replies: Reply[], extra: Partial<SessionMonitorOptions> = {}) {
  const requests: Array<{ url: string; auth: string }> = [];
  const sleeps: number[] = [];
  const waiting: Array<() => void> = [];
  const ended: YoobError[] = [];
  const grants: string[] = [];
  let renewals = 0;
  let token = "st_1";
  const monitor = new SessionMonitor({
    apiBase: () => "https://api2.yoob.com/",
    sessionToken: () => token,
    intervalSeconds: 15,
    renew: async () => { renewals += 1; token = `st_${renewals + 1}`; },
    onGrant: (grant) => grants.push(grant),
    onEnded: (error) => ended.push(error),
    retryDelayMs: (n) => n * 1000,
    sleep: (ms) => { sleeps.push(ms); return new Promise((resolve) => waiting.push(resolve)); },
    fetch: (async (url: string, init: RequestInit) => {
      requests.push({ url, auth: (init.headers as Record<string, string>).authorization });
      const next = replies.shift() ?? { status: 200, body: { stop: false } };
      if (next instanceof Error) throw next;
      return new Response(next.body === undefined ? null : JSON.stringify(next.body), { status: next.status });
    }) as typeof fetch,
    ...extra,
  });
  /** Lets every pending sleep finish, then lets the resulting work settle. */
  const tick = async () => {
    for (const resolve of waiting.splice(0)) resolve();
    for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve));
  };
  /** Runs a beat to completion, moving the fake clock for any retry waits. */
  const drive = async (work: Promise<void>) => {
    let done = false;
    void work.finally(() => { done = true; });
    for (let i = 0; i < 50 && !done; i += 1) await tick();
    assert.ok(done, "the beat finished");
    await work;
  };
  return {
    monitor, requests, sleeps, ended, grants, tick, drive,
    get renewals() { return renewals; },
    setRenew: (fn: () => Promise<void>) => { (monitor as unknown as { options: SessionMonitorOptions }).options.renew = fn; },
  };
}

test("beats on the session's interval from the start, with the session token", async () => {
  const h = harness([]);
  h.monitor.start();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(h.sleeps, [15_000]);
  assert.equal(h.requests.length, 0, "waits one interval before the first beat");
  await h.tick();
  assert.equal(h.requests.length, 1);
  await h.tick();
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[0].url, "https://api2.yoob.com/api/v1/sessions/heartbeat");
  assert.equal(h.requests[0].auth, "Bearer st_1");
  assert.deepEqual(h.ended, []);
  h.monitor.stop();
});

test("ends the session after three consecutive failed heartbeats, retrying with backoff", async () => {
  const h = harness([new TypeError("offline"), { status: 503 }, { status: 409 }]);
  h.monitor.start();
  await h.drive(h.monitor.beatNow());
  assert.equal(h.requests.length, 3);
  assert.deepEqual(h.sleeps.slice(1), [1000, 2000], "backs off between retries");
  assert.equal(h.ended.length, 1);
  assert.equal(h.ended[0].code, "session-ended");
  assert.equal(h.monitor.active, false);
});

test("a successful heartbeat resets the failure count", async () => {
  const h = harness([new TypeError("offline"), new TypeError("offline"), { status: 200, body: {} },
    new TypeError("offline"), new TypeError("offline"), { status: 200, body: {} }]);
  h.monitor.start();
  await h.drive(h.monitor.beatNow());
  assert.equal(h.monitor.consecutiveFailures, 0);
  await h.drive(h.monitor.beatNow());
  assert.equal(h.requests.length, 6);
  assert.deepEqual(h.ended, []);
  h.monitor.stop();
});

test("stops at once when the API refuses the session or the workspace is out of credit", async () => {
  const cases: Array<[Reply, YoobError["code"]]> = [
    [{ status: 401, body: { error: "Missing session token" } }, "unauthorized"],
    [{ status: 403, body: {} }, "unauthorized"],
    [{ status: 402, body: { code: "quota_exceeded" } }, "out-of-credit"],
    [{ status: 200, body: { stop: true, reason: "out-of-credits" } }, "out-of-credit"],
    [{ status: 200, body: { stop: true, reason: "out-of-credits", code: "monthly_cap_reached" } }, "out-of-credit"],
    [{ status: 200, body: { stop: true, reason: "sandbox-limit", code: "sandbox_limit" } }, "session-ended"],
    [{ status: 402, body: { stop: true, reason: "suspended", code: "suspended" } }, "session-ended"],
    [{ status: 403, body: { stop: true, reason: "key-revoked", code: "key_revoked" } }, "unauthorized"],
    [{ status: 402, body: "not json" }, "out-of-credit"],
  ];
  for (const [reply, code] of cases) {
    const h = harness([reply]);
    h.monitor.start();
    await h.drive(h.monitor.beatNow());
    assert.equal(h.requests.length, 1, code);
    assert.equal(h.ended[0]?.code, code);
    assert.equal(h.monitor.active, false);
  }
});

test("opens a new session when the API has forgotten this one, and stops if that fails", async () => {
  const gone = harness([{ status: 404, body: { error: "Unknown or already-ended session." } }, { status: 200, body: {} }]);
  gone.monitor.start();
  await gone.drive(gone.monitor.beatNow());
  assert.equal(gone.renewals, 1);
  await gone.drive(gone.monitor.beatNow());
  assert.equal(gone.requests[1].auth, "Bearer st_2", "the renewed session token is used");
  assert.deepEqual(gone.ended, []);
  gone.monitor.stop();

  const abandoned = harness([{ status: 200, body: { stop: true, reason: "abandoned" } }]);
  abandoned.setRenew(async () => { throw new Error("backend down"); });
  abandoned.monitor.start();
  await abandoned.drive(abandoned.monitor.beatNow());
  assert.equal(abandoned.ended[0]?.code, "session-ended");

  const broke = harness([{ status: 404 }]);
  broke.setRenew(async () => { throw new YoobError("out-of-credit", "no credit"); });
  broke.monitor.start();
  await broke.drive(broke.monitor.beatNow());
  assert.equal(broke.ended[0]?.code, "out-of-credit");
});

test("hands over a renewed download grant and tolerates its absence", async () => {
  const h = harness([
    { status: 200, body: { stop: false, credits_remaining: 9, billed_seconds: 15, reason: null, download_token: "yg1.a", download_token_expires_at: 1790000000 } },
    { status: 200, body: { stop: false, grant: "yg1.b", grant_expires_at: "2026-09-17T13:00:00Z" } },
    { status: 200, body: { stop: false, download_token: "yg1.c", grant: "yg1.ignored" } },
    { status: 200, body: { stop: false, download_token: null, grant: null } },
    { status: 200, body: { stop: false } },
    { status: 200, body: "not an object" },
  ]);
  h.monitor.start();
  for (let i = 0; i < 6; i += 1) await h.drive(h.monitor.beatNow());
  assert.deepEqual(h.grants, ["yg1.a", "yg1.b", "yg1.c"]);
  assert.deepEqual(h.ended, []);
  h.monitor.stop();
});

test("does nothing after stop()", async () => {
  const h = harness([]);
  h.monitor.start();
  h.monitor.stop();
  await h.drive(h.monitor.beatNow());
  await h.tick(); await h.tick();
  assert.equal(h.requests.length, 0);
  assert.deepEqual(h.ended, []);
});

test("fails closed without a session token", async () => {
  const h = harness([], { sessionToken: () => undefined });
  h.monitor.start();
  await h.drive(h.monitor.beatNow());
  assert.equal(h.requests.length, 0);
  assert.equal(h.ended[0]?.code, "unauthorized");
});
