/**
 * FOC Model Upload Script (multi-part)
 *
 * Uploads a local model directory to Filecoin Onchain Cloud via Synapse SDK v0.40+.
 * The model is packed as tar.gz and split into <900MB parts (SP limit is 1016 MiB).
 * Each part is uploaded separately and all CIDs are printed.
 *
 * Usage:
 *   node upload.js --model-dir /path/to/model [--name "Qwen/Qwen2.5-1.5B-Instruct"]
 */

import 'dotenv/config';
// Node 20 polyfill: @filoz/synapse-core uses Promise.withResolvers (Node 22+).
if (!Promise.withResolvers) {
  Promise.withResolvers = function () {
    let resolve, reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
  };
}
import { Synapse } from '@filoz/synapse-sdk';
import { calibration, mainnet } from '@filoz/synapse-core/chains';
import { http } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';
import { statSync, createReadStream, readdirSync, readFileSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';
import { resolve, basename, join } from 'path';
import { tmpdir } from 'os';

const PRIVATE_KEY = process.env.FOC_PRIVATE_KEY;
const NETWORK = process.argv.includes('--network') ? process.argv[process.argv.indexOf('--network')+1] : 'calibration';
const CHAIN = NETWORK === 'mainnet' ? mainnet : calibration;
const RPC_URL = process.env.FOC_RPC_URL || (NETWORK === 'mainnet' ? 'https://api.node.glif.io/rpc/v1' : 'https://api.calibration.node.glif.io/rpc/v1');
if (NETWORK === 'mainnet' && RPC_URL.includes('calibration')) {
  console.error('REFUSING: mainnet upload with a calibration RPC. Pass FOC_RPC_URL=https://api.node.glif.io/rpc/v1 explicitly.');
  process.exit(1);
}
if (NETWORK !== 'mainnet' && RPC_URL.includes('api.node.glif.io')) {
  console.error('REFUSING: calibration upload with a MAINNET RPC — would sign real-chain transactions from a test flow.');
  process.exit(1);
}
const MAX_PART_SIZE = 900 * 1024 * 1024; // 900 MB per part
// Attempts per part. The upload itself is reliable; the on-chain commit that
// follows it times out sporadically on the provider side, so a retry usually
// lands. Backoff grows 30s/60s/... between attempts.
const PART_UPLOAD_ATTEMPTS = Number(process.env.FOC_PART_ATTEMPTS || 4);
// Where the weights volume is mounted; the CID registry is written here so it
// survives log rotation and container restarts.
const MODELS_DIR = process.env.MODEL_CACHE_DIR || '/models';

if (!PRIVATE_KEY) {
  console.error('ERROR: FOC_PRIVATE_KEY is required. Set it in .env');
  process.exit(1);
}

function parseArgs() {
  const args = process.argv.slice(2);
  const result = { modelDir: null, name: null };

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--model-dir' && args[i + 1]) {
      result.modelDir = resolve(args[++i]);
    } else if (args[i] === '--name' && args[i + 1]) {
      result.name = args[++i];
    }
  }

  if (!result.modelDir) {
    console.error('Usage: node upload.js --model-dir /path/to/model [--name "model-name"]');
    process.exit(1);
  }

  if (!result.name) {
    result.name = basename(result.modelDir);
  }

  return result;
}

async function readLargeFile(filePath) {
  const chunks = [];
  const stream = createReadStream(filePath);
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

async function main() {
  const { modelDir, name } = parseArgs();

  try {
    const stat = statSync(modelDir);
    if (!stat.isDirectory()) {
      console.error(`ERROR: ${modelDir} is not a directory`);
      process.exit(1);
    }
  } catch {
    console.error(`ERROR: ${modelDir} does not exist`);
    process.exit(1);
  }

  console.log(`Model directory: ${modelDir}`);
  console.log(`Model name: ${name}`);

  // Pack model directory as tar.gz
  const tarPath = join(tmpdir(), `foc-upload-${basename(modelDir)}.tar.gz`);
  console.log(`\nPacking model to ${tarPath}...`);
  execSync(`tar -czf "${tarPath}" -C "${modelDir}" .`, { stdio: 'inherit' });

  const tarStat = statSync(tarPath);
  const sizeMB = (tarStat.size / (1024 * 1024)).toFixed(1);
  console.log(`Archive size: ${sizeMB} MB`);

  // Split into parts if needed
  const splitDir = join(tmpdir(), `foc-split-${Date.now()}`);
  execSync(`mkdir -p "${splitDir}"`);

  if (tarStat.size > MAX_PART_SIZE) {
    const partSizeMB = Math.floor(MAX_PART_SIZE / (1024 * 1024));
    console.log(`\nArchive exceeds ${partSizeMB}MB limit, splitting into parts...`);
    execSync(`split -b ${partSizeMB}m "${tarPath}" "${splitDir}/part_"`, { stdio: 'inherit' });
    execSync(`rm -f "${tarPath}"`);
  } else {
    execSync(`mv "${tarPath}" "${splitDir}/part_aa"`);
  }

  const parts = readdirSync(splitDir).sort();
  console.log(`Split into ${parts.length} parts`);

  // Initialize Synapse SDK
  console.log(`\nInitializing Synapse SDK (RPC: ${RPC_URL})...`);
  const account = privateKeyToAccount(PRIVATE_KEY);
  console.log(`Account: ${account.address}`);

  const synapse = Synapse.create({
    account,
    chain: CHAIN,
    transport: http(RPC_URL),
  });
  console.log(`Synapse SDK initialized (network ${NETWORK}, chain ${CHAIN.id}, view ${CHAIN.contracts.fwssView.address})`);

  // Upload each part
  const results = [];
  const startTime = Date.now();

  for (let i = 0; i < parts.length; i++) {
    const partPath = join(splitDir, parts[i]);
    const partStat = statSync(partPath);
    const partMB = (partStat.size / (1024 * 1024)).toFixed(1);

    console.log(`\n--- Part ${i + 1}/${parts.length}: ${parts[i]} (${partMB} MB) ---`);
    console.log('Reading...');
    const data = await readLargeFile(partPath);

    console.log('Uploading to FOC...');
    const partStart = Date.now();
    // Retry each part on its own. The bytes reach the storage provider quickly;
    // what times out is the on-chain commit ("data is stored but not on-chain"),
    // and it fails at random parts. Without this, one flaky commit discards the
    // whole model — a 40-part upload died that way, hours in.
    let result;
    for (let attempt = 1; ; attempt++) {
      try {
        result = await synapse.storage.upload(data);
        break;
      } catch (err) {
        const msg = err.shortMessage || err.message || String(err);
        if (attempt >= PART_UPLOAD_ATTEMPTS) {
          console.error(`Part ${i + 1} failed after ${attempt} attempts: ${msg}`);
          throw err;
        }
        const backoffSec = 30 * attempt;
        console.warn(`Part ${i + 1} attempt ${attempt}/${PART_UPLOAD_ATTEMPTS} failed: ${msg}`);
        console.warn(`  retrying in ${backoffSec}s ...`);
        await new Promise((r) => setTimeout(r, backoffSec * 1000));
      }
    }
    const partElapsed = ((Date.now() - partStart) / 1000).toFixed(1);

    console.log(`Part ${i + 1} uploaded in ${partElapsed}s`);
    console.log(`Result:`, result);
    // Keep the piece CID itself, not String(result) — that stringifies to
    // "[object Object]", which is how one model's CIDs became unrecoverable
    // once its log was overwritten. Without the CIDs the bytes are on-chain
    // but unreachable.
    results.push({
      part: parts[i],
      pieceCid: String(result?.pieceCid ?? result),
      sizeBytes: partStat.size,
    });
  }

  const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  console.log(`\n========================================`);
  console.log(`Upload complete in ${totalElapsed}s!`);
  console.log(`Model name: ${name}`);
  console.log(`Parts: ${parts.length}`);
  console.log(`Results:`);
  for (const r of results) {
    console.log(`  ${r.part}: ${r.pieceCid}`);
  }

  // Persist the CIDs next to the weights. Logs get truncated by the next run;
  // this registry is what a downloader actually needs (ordered CIDs — the parts
  // must be concatenated in upload order or the archive is garbage).
  const registryPath = join(MODELS_DIR, 'foc-registry.json');
  let registry = {};
  try { registry = JSON.parse(readFileSync(registryPath, 'utf8')); } catch { }
  registry[name] = {
    network: NETWORK,
    chainId: CHAIN.id,
    parts: results.map((r) => r.pieceCid),
    partSizes: results.map((r) => r.sizeBytes),
    totalBytes: results.reduce((a, r) => a + r.sizeBytes, 0),
    uploadedAt: new Date().toISOString(),
  };
  writeFileSync(registryPath, JSON.stringify(registry, null, 2) + '\n');
  console.log(`Registry updated: ${registryPath}`);
  console.log(`========================================`);

  // Cleanup
  execSync(`rm -rf "${splitDir}"`);
}

main().catch((err) => {
  console.error('\nUpload failed:', err);
  process.exit(1);
});
