/**
 * FOC Bridge Server
 *
 * HTTP bridge between the Python inference service and the Filecoin Onchain Cloud.
 * Downloads model parts directly from SP retrieval URLs.
 *
 * Endpoints:
 *   GET  /health                    - Health check
 *   POST /resolve                   - Check if piece CID exists on SP
 *   POST /download-model            - Download multi-part model to disk, return path
 *
 * Hardening (see m1-fixes doc F1-F4):
 *   - destPath is confined to FOC_MODELS_DIR (no arbitrary file write / traversal)
 *   - pieceCids are charset-validated (no path injection into the SP URL)
 *   - optional bearer auth (FOC_BRIDGE_TOKEN)
 *   - per-part fetch timeout, max parts, and max total size caps
 */

import 'dotenv/config';
import express from 'express';
import path from 'path';
import { createWriteStream } from 'fs';
import { createHash } from 'crypto';
import { stat, unlink, mkdir } from 'fs/promises';
import { pipeline } from 'stream/promises';
import { Readable, Transform } from 'stream';
import { fileURLToPath } from 'url';

const PORT = parseInt(process.env.FOC_BRIDGE_PORT || '3100', 10);

// SP retrieval base URLs — configurable via environment variables
const SP_URLS = {
  primary: process.env.SP_URL_PRIMARY || 'https://caliberation-pdp.infrafolio.com/piece',
  secondary: process.env.SP_URL_SECONDARY || 'https://calib2.ezpdpz.net/piece',
};

// Downloads are confined to this base directory (the shared models volume).
const MODELS_DIR = path.resolve(process.env.FOC_MODELS_DIR || '/models');
// Optional bearer token protecting /resolve and /download-model.
const AUTH_TOKEN = process.env.FOC_BRIDGE_TOKEN || '';
// Resource caps.
const MAX_PARTS = parseInt(process.env.FOC_MAX_PARTS || '512', 10);
const FETCH_TIMEOUT_MS = parseInt(process.env.FOC_FETCH_TIMEOUT_MS || '300000', 10); // 5 min/part
const MAX_TOTAL_BYTES = parseInt(process.env.FOC_MAX_TOTAL_BYTES || String(200 * 1024 * 1024 * 1024), 10); // 200 GB

// --- Pure helpers (exported for unit tests) ---

/** Validate a piece CID is safe to interpolate into a URL path segment. */
export function isValidCid(cid) {
  return typeof cid === 'string' && cid.length > 0 && cid.length <= 256 && /^[A-Za-z0-9]+$/.test(cid);
}

/**
 * Resolve destPath and ensure it stays within baseDir. Throws on traversal.
 * Returns the resolved absolute path.
 */
export function resolveSafeDest(destPath, baseDir = MODELS_DIR) {
  if (typeof destPath !== 'string' || destPath.length === 0) {
    throw new Error('destPath must be a non-empty string');
  }
  const base = path.resolve(baseDir);
  const resolved = path.resolve(base, destPath);
  if (resolved !== base && !resolved.startsWith(base + path.sep)) {
    throw new Error('destPath escapes the allowed base directory');
  }
  return resolved;
}

/** fetch with an abort timeout. */
async function fetchWithTimeout(url, opts = {}) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(t);
  }
}

export function createApp(opts = {}) {
  const modelsDir = path.resolve(opts.modelsDir || MODELS_DIR);
  const authToken = opts.authToken !== undefined ? opts.authToken : AUTH_TOKEN;
  const maxParts = opts.maxParts || MAX_PARTS;
  const maxTotalBytes = opts.maxTotalBytes || MAX_TOTAL_BYTES;
  const spUrls = opts.spUrls || SP_URLS;
  // Injectable fetch (url, opts) => Response — defaults to the real timeout-wrapped fetch.
  const doFetch = opts.fetchImpl || fetchWithTimeout;
  // Guards against two concurrent downloads writing the same destination file.
  const inFlight = new Set();

  // Fetch one part, falling back to the secondary SP on ANY primary failure —
  // a thrown error (network/timeout) OR a non-ok HTTP status (e.g. the piece is
  // not stored on the primary SP). Previously only thrown errors triggered fallback.
  async function fetchPart(cid) {
    const primaryUrl = `${spUrls.primary}/${cid}`;
    try {
      const resp = await doFetch(primaryUrl);
      if (resp.ok) return resp;
      console.log(`  Primary SP returned ${resp.status}, trying secondary...`);
    } catch (err) {
      console.log(`  Primary SP failed (${err.message}), trying secondary...`);
    }
    return await doFetch(`${spUrls.secondary}/${cid}`);
  }

  // Probe one piece for availability/size with a 0-0 range request, falling back
  // to the secondary SP on ANY primary failure (a thrown error OR a non-ok,
  // non-206 status) — mirrors fetchPart so /resolve survives a primary SP that is
  // down or simply doesn't hold the piece. Previously /resolve only tried the
  // primary, so a flaky/unreachable primary made resolve 500 even when the
  // secondary held the piece.
  async function headPiece(cid) {
    const opts = { headers: { Range: 'bytes=0-0' } };
    try {
      const resp = await doFetch(`${spUrls.primary}/${cid}`, opts);
      if (resp.ok || resp.status === 206) return resp;
      console.log(`  Primary SP returned ${resp.status} on resolve, trying secondary...`);
    } catch (err) {
      console.log(`  Primary SP failed on resolve (${err.message}), trying secondary...`);
    }
    return await doFetch(`${spUrls.secondary}/${cid}`, opts);
  }

  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // --- Auth middleware (opt-in) ---
  function requireAuth(req, res, next) {
    if (!authToken) return next();
    const auth = req.headers['authorization'] || '';
    if (auth !== `Bearer ${authToken}`) {
      return res.status(401).json({ error: 'invalid or missing Authorization' });
    }
    next();
  }

  // --- Health check (open) ---
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', spUrls: SP_URLS, modelsDir });
  });

  // --- Resolve: check if piece exists on SP ---
  app.post('/resolve', requireAuth, async (req, res) => {
    const { pieceCid } = req.body;
    if (!isValidCid(pieceCid)) {
      return res.status(400).json({ error: 'valid pieceCid is required' });
    }

    try {
      const resp = await headPiece(pieceCid);
      if (resp.ok || resp.status === 206) {
        const range = resp.headers.get('content-range') || '';
        const match = range.match(/\/(\d+)$/);
        const size = match ? parseInt(match[1], 10) : 0;
        res.json({ pieceCid, available: true, sizeBytes: size });
      } else {
        res.status(404).json({ pieceCid, available: false, error: `SP returned ${resp.status}` });
      }
    } catch (err) {
      console.error(`Resolve error for ${pieceCid}:`, err.message);
      res.status(500).json({ error: err.message });
    }
  });

  // --- Download multi-part model ---
  app.post('/download-model', requireAuth, async (req, res) => {
    const { pieceCids, destPath, sha256 } = req.body;
    // A2 weight-integrity: sha256 (64 hex chars) is the expected digest of the FINAL
    // assembled file, pinned in the model catalog. Retrieval endpoints are untrusted:
    // a malicious/compromised SP could serve tampered (backdoored/degraded) weights,
    // and piece CIDs are never actually recomputed here — this digest is the gate.
    if (sha256 !== undefined && !/^[0-9a-fA-F]{64}$/.test(String(sha256))) {
      return res.status(400).json({ error: 'sha256 must be 64 hex chars' });
    }
    if (!pieceCids || !Array.isArray(pieceCids) || pieceCids.length === 0) {
      return res.status(400).json({ error: 'pieceCids array is required' });
    }
    if (pieceCids.length > maxParts) {
      return res.status(400).json({ error: `too many parts (${pieceCids.length} > ${maxParts})` });
    }
    for (const cid of pieceCids) {
      if (!isValidCid(cid)) {
        return res.status(400).json({ error: `invalid pieceCid: ${String(cid).slice(0, 40)}` });
      }
    }

    // Confine output to the models volume (reject traversal / arbitrary write).
    let outPath;
    try {
      outPath = resolveSafeDest(destPath || path.join(modelsDir, 'foc-model-download.tar.gz'), modelsDir);
    } catch (err) {
      return res.status(400).json({ error: err.message });
    }

    // Reject a second concurrent download to the same destination (would interleave
    // writes and corrupt the file).
    if (inFlight.has(outPath)) {
      return res.status(409).json({ error: 'a download to this destination is already in progress' });
    }
    inFlight.add(outPath);

    console.log(`Model download requested: ${pieceCids.length} parts -> ${outPath}`);
    const startTime = Date.now();

    try {
      // Ensure the destination's parent directory exists before opening the
      // write stream. Otherwise createWriteStream emits an async 'error' (ENOENT)
      // that, with no listener attached yet, crashes the whole process.
      await mkdir(path.dirname(outPath), { recursive: true });
      const ws = createWriteStream(outPath);
      let totalSize = 0;
      const hasher = createHash('sha256'); // digest of the assembled file, streamed

      for (let i = 0; i < pieceCids.length; i++) {
        const cid = pieceCids[i];
        console.log(`  Downloading part ${i + 1}/${pieceCids.length}: ${cid.substring(0, 30)}...`);

        const resp = await fetchPart(cid);

        if (!resp.ok) {
          ws.destroy();
          throw new Error(`SP returned ${resp.status} for part ${i + 1} (${cid})`);
        }

        const sizeBefore = ws.bytesWritten;
        const hashTap = new Transform({
          transform(chunk, _enc, cb) { hasher.update(chunk); cb(null, chunk); },
        });
        await pipeline(Readable.fromWeb(resp.body), hashTap, ws, { end: false });
        totalSize = ws.bytesWritten;

        if (totalSize > maxTotalBytes) {
          ws.destroy();
          throw new Error(`download exceeded max size (${maxTotalBytes} bytes)`);
        }
        const partSize = ws.bytesWritten - sizeBefore;
        console.log(`  Part ${i + 1} done: ${(partSize / (1024 * 1024)).toFixed(1)} MB`);
      }

      ws.end();
      await new Promise((resolve) => ws.on('finish', resolve));

      const finalStat = await stat(outPath);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const computed = hasher.digest('hex');

      // A2: verify BEFORE anything can load these bytes. On mismatch the file is
      // deleted (never leave poisoned weights on the shared volume) and the caller
      // gets a hard 422. Without an expected digest the download is UNVERIFIED --
      // allowed for backward compat, but said loudly.
      if (sha256 !== undefined && computed !== String(sha256).toLowerCase()) {
        try { await unlink(outPath); } catch {}
        console.error(`WEIGHT INTEGRITY FAILURE: expected sha256 ${sha256}, got ${computed} -- file deleted`);
        return res.status(422).json({
          error: 'sha256 mismatch: downloaded weights do NOT match the pinned digest (file deleted)',
          expected: String(sha256).toLowerCase(),
          computed,
        });
      }
      if (sha256 === undefined) {
        console.warn(`UNVERIFIED model download (no expected sha256 supplied): ${outPath} sha256=${computed}`);
      }
      console.log(`Model download complete: ${pieceCids.length} parts, ${(finalStat.size / (1024 * 1024)).toFixed(1)} MB, ${elapsed}s, sha256=${computed} verified=${sha256 !== undefined}`);

      res.json({
        success: true,
        path: outPath,
        sizeBytes: finalStat.size,
        parts: pieceCids.length,
        elapsedSec: parseFloat(elapsed),
        sha256: computed,
        verified: sha256 !== undefined,
      });
    } catch (err) {
      console.error('Model download error:', err.message);
      try { await unlink(outPath); } catch {}
      res.status(500).json({ error: err.message });
    } finally {
      inFlight.delete(outPath);
    }
  });

  return app;
}

// --- Start server only when run directly (not when imported by tests) ---
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (isMain) {
  const app = createApp();
  app.listen(PORT, '127.0.0.1', () => {
    console.log(`FOC Bridge listening on http://127.0.0.1:${PORT}`);
    console.log('SP URLs:', SP_URLS);
    console.log('Models dir:', MODELS_DIR);
    console.log('Auth:', AUTH_TOKEN ? 'enabled' : 'disabled');
  });
}
