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
 */

import 'dotenv/config';
import express from 'express';
import { createWriteStream } from 'fs';
import { stat, unlink } from 'fs/promises';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

const PORT = parseInt(process.env.FOC_BRIDGE_PORT || '3100', 10);

// SP retrieval base URLs — configurable via environment variables
const SP_URLS = {
  primary: process.env.SP_URL_PRIMARY || 'https://caliberation-pdp.infrafolio.com/piece',
  secondary: process.env.SP_URL_SECONDARY || 'https://calib2.ezpdpz.net/piece',
};

const app = express();
app.use(express.json());

// --- Health check ---
app.get('/health', (req, res) => {
  res.json({ status: 'ok', spUrls: SP_URLS });
});

// --- Resolve: check if piece exists on SP ---
app.post('/resolve', async (req, res) => {
  const { pieceCid } = req.body;
  if (!pieceCid) {
    return res.status(400).json({ error: 'pieceCid is required' });
  }

  try {
    const url = `${SP_URLS.primary}/${pieceCid}`;
    // SP doesn't support HEAD, use GET with Range to check availability
    const resp = await fetch(url, { headers: { 'Range': 'bytes=0-0' } });
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

// --- Download a single part to disk, streaming ---
async function downloadPart(cid, destPath) {
  const url = `${SP_URLS.primary}/${cid}`;
  let resp;
  try {
    resp = await fetch(url);
  } catch (err) {
    console.log(`  Primary SP failed, trying secondary...`);
    resp = await fetch(`${SP_URLS.secondary}/${cid}`);
  }

  if (!resp.ok) {
    throw new Error(`SP returned ${resp.status} for ${cid}`);
  }

  const ws = createWriteStream(destPath);
  await pipeline(Readable.fromWeb(resp.body), ws);

  const info = await stat(destPath);
  return info.size;
}

// --- Download multi-part model ---
// Body: { "pieceCids": ["cid1", "cid2", "cid3"], "destPath": "/tmp/model.tar.gz" }
// Downloads each part from SP, concatenates to destPath
app.post('/download-model', async (req, res) => {
  const { pieceCids, destPath } = req.body;
  if (!pieceCids || !Array.isArray(pieceCids) || pieceCids.length === 0) {
    return res.status(400).json({ error: 'pieceCids array is required' });
  }
  const outPath = destPath || '/tmp/foc-model-download.tar.gz';

  console.log(`Model download requested: ${pieceCids.length} parts -> ${outPath}`);
  const startTime = Date.now();

  try {
    // Download parts and concatenate by appending to output file
    const ws = createWriteStream(outPath);
    let totalSize = 0;

    for (let i = 0; i < pieceCids.length; i++) {
      const cid = pieceCids[i];
      const url = `${SP_URLS.primary}/${cid}`;
      console.log(`  Downloading part ${i + 1}/${pieceCids.length}: ${cid.substring(0, 30)}...`);

      const partStart = Date.now();
      let resp;
      try {
        resp = await fetch(url);
      } catch (err) {
        console.log(`  Primary SP failed, trying secondary...`);
        resp = await fetch(`${SP_URLS.secondary}/${cid}`);
      }

      if (!resp.ok) {
        ws.destroy();
        throw new Error(`SP returned ${resp.status} for part ${i + 1} (${cid})`);
      }

      // Track bytes written through the stream
      const sizeBefore = ws.bytesWritten;
      await pipeline(Readable.fromWeb(resp.body), ws, { end: false });
      const partSize = ws.bytesWritten - sizeBefore;
      const partElapsed = ((Date.now() - partStart) / 1000).toFixed(1);

      totalSize += partSize;
      const speed = (partSize / (1024 * 1024)) / (parseFloat(partElapsed) || 1);
      console.log(`  Part ${i + 1} done: ${(partSize / (1024 * 1024)).toFixed(1)} MB in ${partElapsed}s (${speed.toFixed(1)} MB/s)`);
    }

    ws.end();
    await new Promise((resolve) => ws.on('finish', resolve));

    const finalStat = await stat(outPath);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`Model download complete: ${pieceCids.length} parts, ${(finalStat.size / (1024 * 1024)).toFixed(1)} MB, ${elapsed}s`);

    res.json({
      success: true,
      path: outPath,
      sizeBytes: finalStat.size,
      parts: pieceCids.length,
      elapsedSec: parseFloat(elapsed),
    });
  } catch (err) {
    console.error('Model download error:', err.message);
    // Clean up partial file
    try { await unlink(outPath); } catch {}
    res.status(500).json({ error: err.message });
  }
});

// --- Start server ---
app.listen(PORT, '127.0.0.1', () => {
  console.log(`FOC Bridge listening on http://127.0.0.1:${PORT}`);
  console.log('SP URLs:', SP_URLS);
  console.log('Endpoints:');
  console.log('  GET  /health');
  console.log('  POST /resolve          { "pieceCid": "..." }');
  console.log('  POST /download-model   { "pieceCids": ["..."], "destPath": "/tmp/..." }');
});
