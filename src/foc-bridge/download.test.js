/**
 * Download-engine tests for the FOC bridge: multi-part assembly, secondary-SP
 * fallback (incl. the non-ok-status fix), partial-failure cleanup, size cap, and
 * the concurrent-download lock. Run: node --test
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, existsSync, readFileSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { ReadableStream } from 'node:stream/web';

import { createApp } from './server.js';

// --- helpers ---
function listen(app) {
  return new Promise((resolve) => {
    const server = app.listen(0, '127.0.0.1', () => resolve(server));
  });
}
function req(server, method, p, { token, body } = {}) {
  const addr = server.address();
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const r = http.request({ host: addr.address, port: addr.port, method, path: p, headers }, (res) => {
      let buf = '';
      res.on('data', (c) => (buf += c));
      res.on('end', () => resolve({ status: res.statusCode, body: buf ? JSON.parse(buf) : null }));
    });
    r.on('error', reject);
    if (data) r.write(data);
    r.end();
  });
}
function webBody(buf) {
  return new ReadableStream({
    start(c) { c.enqueue(new Uint8Array(buf)); c.close(); },
  });
}
function okResp(buf) { return { ok: true, status: 200, body: webBody(Buffer.from(buf)) }; }
function tmpDir() { return mkdtempSync(path.join(os.tmpdir(), 'foc-dl-')); }
const SP = { primary: 'http://primary.test/piece', secondary: 'http://secondary.test/piece' };

// === multi-part assembly happy path ===
test('assembles multiple parts in order into one file', async () => {
  const dir = tmpDir();
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    if (url.endsWith('/p0')) return okResp('AAA');
    if (url.endsWith('/p1')) return okResp('BBB');
    if (url.endsWith('/p2')) return okResp('CCC');
    return { ok: false, status: 404 };
  };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'model.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['p0', 'p1', 'p2'], destPath: out } });
    assert.equal(r.status, 200);
    assert.equal(r.body.success, true);
    assert.equal(r.body.parts, 3);
    assert.equal(readFileSync(out, 'utf8'), 'AAABBBCCC'); // assembled in order
    assert.equal(r.body.sizeBytes, 9);
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === REGRESSION: secondary-SP fallback now triggers on a non-ok primary status ===
test('falls back to secondary SP when primary returns a non-ok status', async () => {
  const dir = tmpDir();
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    if (url.startsWith(SP.primary)) return { ok: false, status: 404 }; // primary does not have the piece
    return okResp('SECONDARY-DATA'); // secondary serves it
  };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'm.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r.status, 200, 'should succeed via secondary');
    assert.equal(readFileSync(out, 'utf8'), 'SECONDARY-DATA');
    // both SPs were contacted — proving the fallback fired on a non-ok status (the bug fix)
    assert.ok(calls.some((u) => u.startsWith(SP.primary)), 'primary attempted');
    assert.ok(calls.some((u) => u.startsWith(SP.secondary)), 'secondary attempted');
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === fallback also triggers on a thrown (network) error ===
test('falls back to secondary SP when primary throws', async () => {
  const dir = tmpDir();
  const fetchImpl = async (url) => {
    if (url.startsWith(SP.primary)) throw new Error('ECONNREFUSED');
    return okResp('VIA-SECONDARY');
  };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'm.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r.status, 200);
    assert.equal(readFileSync(out, 'utf8'), 'VIA-SECONDARY');
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === both SPs fail → 500 + partial file cleaned up ===
test('cleans up the partial file when a part fails on both SPs', async () => {
  const dir = tmpDir();
  const fetchImpl = async (url) => {
    if (url.endsWith('/good')) return okResp('PART-ONE');
    return { ok: false, status: 500 }; // 'bad' fails on both primary and secondary
  };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'm.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['good', 'bad'], destPath: out } });
    assert.equal(r.status, 500);
    assert.match(r.body.error, /SP returned 500/);
    assert.equal(existsSync(out), false, 'partial file must be unlinked on failure');
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === total-size cap enforced ===
test('rejects when total download exceeds the size cap', async () => {
  const dir = tmpDir();
  // Part must exceed the fs write-buffer (16 KiB) so bytes flush to disk and
  // ws.bytesWritten reflects the overage that trips the cap.
  const fetchImpl = async () => okResp('X'.repeat(256 * 1024)); // 256 KiB
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl, maxTotalBytes: 1024 });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'm.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['big'], destPath: out } });
    assert.equal(r.status, 500);
    assert.match(r.body.error, /exceeded max size/);
    assert.equal(existsSync(out), false);
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === concurrent download to the same destPath → 409 ===
test('rejects a concurrent download to the same destPath', async () => {
  const dir = tmpDir();
  let release;
  const gate = new Promise((res) => { release = res; });
  let firstStartedResolve;
  const firstStarted = new Promise((res) => { firstStartedResolve = res; });
  const fetchImpl = async (url) => {
    firstStartedResolve();
    await gate; // hold the first request's fetch open
    return okResp('DATA');
  };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'same.tar.gz');
    const p1 = req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    await firstStarted; // ensure req1 holds the in-flight lock
    const r2 = await req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r2.status, 409, 'second concurrent download to same dest must be rejected');
    assert.match(r2.body.error, /already in progress/);
    release();
    const r1 = await p1;
    assert.equal(r1.status, 200);
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === non-existent parent dir → auto-created, download succeeds (no process crash) ===
// Regression for a real bug found on the server: createWriteStream into a missing
// directory emitted an unhandled 'error' (ENOENT) that crashed the whole process.
test('auto-creates the destination parent dir instead of crashing', async () => {
  const dir = tmpDir();
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl: async () => okResp('DATA') });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'no', 'such', 'dir', 'm.tar.gz'); // parents don't exist
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r.status, 200);
    assert.equal(readFileSync(out, 'utf8'), 'DATA'); // dir created + file written
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// === missing / empty / non-array pieceCids → 400 ===
test('rejects missing/empty/non-array pieceCids', async () => {
  const app = createApp({ authToken: '', modelsDir: os.tmpdir(), spUrls: SP, fetchImpl: async () => okResp('x') });
  const server = await listen(app);
  try {
    for (const body of [{}, { pieceCids: [] }, { pieceCids: 'abc' }]) {
      const r = await req(server, 'POST', '/download-model', { body });
      assert.equal(r.status, 400, `body ${JSON.stringify(body)} should be 400`);
    }
  } finally {
    server.close();
  }
});

// === auth ACCEPTANCE: a valid token proceeds past the auth gate ===
test('accepts a valid bearer token (auth pass-through)', async () => {
  const dir = tmpDir();
  const app = createApp({ authToken: 'secret', modelsDir: dir, spUrls: SP, fetchImpl: async () => okResp('DATA') });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'm.tar.gz');
    const r = await req(server, 'POST', '/download-model', { token: 'secret', body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r.status, 200, 'valid token must NOT be rejected');
  } finally {
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- helper: a ranged-probe response carrying a content-range header (for /resolve) ---
function rangeResp(status, total) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k) => (k.toLowerCase() === 'content-range' ? `bytes 0-0/${total}` : null) },
  };
}
const noHdr = (status) => ({ ok: false, status, headers: { get: () => null } });

// === REGRESSION: /resolve falls back to the secondary SP on a NON-OK primary status ===
// Previously /resolve only queried the primary SP, so a primary that was down or did
// not hold the piece made resolve 500/404 even when the secondary had it.
test('/resolve falls back to secondary SP when primary returns a non-ok status', async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    if (url.startsWith(SP.primary)) return noHdr(404); // primary does not have the piece
    return rangeResp(206, 943718400);                  // secondary serves the 0-0 range
  };
  const app = createApp({ authToken: '', modelsDir: os.tmpdir(), spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/resolve', { body: { pieceCid: 'c0' } });
    assert.equal(r.status, 200, 'should resolve via secondary');
    assert.equal(r.body.available, true);
    assert.equal(r.body.sizeBytes, 943718400, 'size parsed from secondary content-range');
    assert.ok(calls.some((u) => u.startsWith(SP.primary)), 'primary attempted');
    assert.ok(calls.some((u) => u.startsWith(SP.secondary)), 'secondary attempted (the fix)');
  } finally {
    server.close();
  }
});

// === /resolve also falls back when the primary THROWS (network/timeout) ===
test('/resolve falls back to secondary SP when primary throws', async () => {
  const fetchImpl = async (url) => {
    if (url.startsWith(SP.primary)) throw new Error('ECONNREFUSED');
    return rangeResp(206, 12345);
  };
  const app = createApp({ authToken: '', modelsDir: os.tmpdir(), spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/resolve', { body: { pieceCid: 'c0' } });
    assert.equal(r.status, 200);
    assert.equal(r.body.available, true);
    assert.equal(r.body.sizeBytes, 12345);
  } finally {
    server.close();
  }
});

// === /resolve reports unavailable (404) only when BOTH SPs return non-ok ===
test('/resolve reports unavailable when both SPs return a non-ok status', async () => {
  const fetchImpl = async () => noHdr(404);
  const app = createApp({ authToken: '', modelsDir: os.tmpdir(), spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/resolve', { body: { pieceCid: 'c0' } });
    assert.equal(r.status, 404);
    assert.equal(r.body.available, false);
  } finally {
    server.close();
  }
});

// --- A2 weight-integrity: sha256 verification of the assembled download ---------

test('A2: matching sha256 verifies (verified:true, digest echoed)', async () => {
  const dir = tmpDir();
  const fetchImpl = async (url) => url.endsWith('/p0') ? okResp('hello ') :
    url.endsWith('/p1') ? okResp('world') : { ok: false, status: 404 };
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP, fetchImpl });
  const server = await listen(app);
  try {
    const expected = createHash('sha256').update('hello world').digest('hex');
    const out = path.join(dir, 'ok.tar.gz');
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['p0', 'p1'], destPath: out, sha256: expected } });
    assert.equal(r.status, 200);
    assert.equal(r.body.verified, true);
    assert.equal(r.body.sha256, expected);
    assert.equal(readFileSync(out, 'utf8'), 'hello world');
  } finally { server.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('A2: mismatched sha256 is REJECTED (422) and the poisoned file deleted', async () => {
  const dir = tmpDir();
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP,
    fetchImpl: async () => okResp('tampered weights!!') });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'bad.tar.gz');
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['c0'], destPath: out, sha256: 'a'.repeat(64) } });
    assert.equal(r.status, 422);
    assert.match(r.body.error, /sha256 mismatch/);
    assert.equal(r.body.computed, createHash('sha256').update('tampered weights!!').digest('hex'));
    assert.equal(existsSync(out), false); // never leave poisoned weights on disk
  } finally { server.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('A2: no sha256 stays backward-compatible (verified:false, digest still returned)', async () => {
  const dir = tmpDir();
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP,
    fetchImpl: async () => okResp('legacy') });
  const server = await listen(app);
  try {
    const out = path.join(dir, 'legacy.tar.gz');
    const r = await req(server, 'POST', '/download-model', { body: { pieceCids: ['c0'], destPath: out } });
    assert.equal(r.status, 200);
    assert.equal(r.body.verified, false);
    assert.equal(r.body.sha256, createHash('sha256').update('legacy').digest('hex'));
  } finally { server.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('A2: malformed sha256 is rejected up front (400)', async () => {
  const dir = tmpDir();
  const app = createApp({ authToken: '', modelsDir: dir, spUrls: SP,
    fetchImpl: async () => okResp('x') });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['c0'], destPath: path.join(dir, 'x'), sha256: 'not-hex' } });
    assert.equal(r.status, 400);
  } finally { server.close(); rmSync(dir, { recursive: true, force: true }); }
});
