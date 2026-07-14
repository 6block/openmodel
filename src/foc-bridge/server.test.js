/**
 * F1-F4 regression tests for the FOC bridge hardening.
 * Run: node --test
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { createApp, resolveSafeDest, isValidCid } from './server.js';

// --- F1: destPath confinement (pure) ---
test('resolveSafeDest accepts paths within base', () => {
  const base = '/models';
  assert.equal(resolveSafeDest('/models/x/y.tar.gz', base), '/models/x/y.tar.gz');
  assert.equal(resolveSafeDest('/models/.tmp/a.tar.gz', base), '/models/.tmp/a.tar.gz');
});

test('resolveSafeDest rejects traversal and out-of-base paths', () => {
  const base = '/models';
  assert.throws(() => resolveSafeDest('/etc/passwd', base));
  assert.throws(() => resolveSafeDest('/models/../etc/passwd', base));
  assert.throws(() => resolveSafeDest('', base));
  assert.throws(() => resolveSafeDest('/models-evil/x', base)); // prefix trick
});

// --- F4: CID validation (pure) ---
test('isValidCid accepts alphanumeric CIDs, rejects injection', () => {
  assert.equal(isValidCid('baga6ea4seaqabc123'), true);
  assert.equal(isValidCid('../../etc/passwd'), false);
  assert.equal(isValidCid('cid/with/slash'), false);
  assert.equal(isValidCid('cid with space'), false);
  assert.equal(isValidCid(''), false);
  assert.equal(isValidCid(null), false);
});

// --- HTTP helpers ---
function listen(app) {
  return new Promise((resolve) => {
    const server = app.listen(0, '127.0.0.1', () => resolve(server));
  });
}

function req(server, method, path, { token, body } = {}) {
  const addr = server.address();
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const r = http.request(
      { host: addr.address, port: addr.port, method, path, headers },
      (res) => {
        let buf = '';
        res.on('data', (c) => (buf += c));
        res.on('end', () => resolve({ status: res.statusCode, body: buf ? JSON.parse(buf) : null }));
      }
    );
    r.on('error', reject);
    if (data) r.write(data);
    r.end();
  });
}

// --- F2: auth ---
test('F2: auth required when token configured; /health stays open', async () => {
  const app = createApp({ authToken: 'secret', modelsDir: os.tmpdir() });
  const server = await listen(app);
  try {
    assert.equal((await req(server, 'GET', '/health')).status, 200);

    const noTok = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['abc'], destPath: path.join(os.tmpdir(), 'x.tar.gz') } });
    assert.equal(noTok.status, 401);

    const wrong = await req(server, 'POST', '/download-model',
      { token: 'nope', body: { pieceCids: ['abc'], destPath: path.join(os.tmpdir(), 'x.tar.gz') } });
    assert.equal(wrong.status, 401);
  } finally {
    server.close();
  }
});

// --- F1 over HTTP: traversal rejected before any fetch ---
test('F1: download-model rejects destPath outside base dir', async () => {
  const app = createApp({ authToken: '', modelsDir: path.join(os.tmpdir(), 'foc-models') });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['validcid123'], destPath: '/etc/cron.d/evil' } });
    assert.equal(r.status, 400);
    assert.match(r.body.error, /base directory/);
  } finally {
    server.close();
  }
});

// --- F3 + F4 over HTTP: limits and cid validation reject before fetch ---
test('F3: too many parts rejected', async () => {
  const app = createApp({ authToken: '', modelsDir: os.tmpdir(), maxParts: 3 });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['a', 'b', 'c', 'd'], destPath: path.join(os.tmpdir(), 'x.tar.gz') } });
    assert.equal(r.status, 400);
    assert.match(r.body.error, /too many parts/);
  } finally {
    server.close();
  }
});

test('F4: invalid cid rejected over HTTP', async () => {
  const app = createApp({ authToken: '', modelsDir: os.tmpdir() });
  const server = await listen(app);
  try {
    const r = await req(server, 'POST', '/download-model',
      { body: { pieceCids: ['../../etc/passwd'], destPath: path.join(os.tmpdir(), 'x.tar.gz') } });
    assert.equal(r.status, 400);
    assert.match(r.body.error, /invalid pieceCid/);
  } finally {
    server.close();
  }
});

test('auth disabled allows access (backward compat)', async () => {
  const app = createApp({ authToken: '', modelsDir: os.tmpdir() });
  const server = await listen(app);
  try {
    // Invalid cid still 400, but NOT 401 — proves auth is not enforced.
    const r = await req(server, 'POST', '/resolve', { body: { pieceCid: 'bad/cid' } });
    assert.equal(r.status, 400);
  } finally {
    server.close();
  }
});
