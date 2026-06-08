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
import { Synapse } from '@filoz/synapse-sdk';
import { calibration } from '@filoz/synapse-core/chains';
import { http } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';
import { statSync, createReadStream, readdirSync } from 'fs';
import { execSync } from 'child_process';
import { resolve, basename, join } from 'path';
import { tmpdir } from 'os';

const PRIVATE_KEY = process.env.FOC_PRIVATE_KEY;
const RPC_URL = process.env.FOC_RPC_URL || 'https://api.calibration.node.glif.io/rpc/v1';
const MAX_PART_SIZE = 900 * 1024 * 1024; // 900 MB per part

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
  const tarPath = join(tmpdir(), `foc-upload-${Date.now()}.tar.gz`);
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
    chain: calibration,
    transport: http(RPC_URL),
  });
  console.log('Synapse SDK initialized');

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
    const result = await synapse.storage.upload(data);
    const partElapsed = ((Date.now() - partStart) / 1000).toFixed(1);

    console.log(`Part ${i + 1} uploaded in ${partElapsed}s`);
    console.log(`Result:`, result);
    results.push({ part: parts[i], result: String(result) });
  }

  const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  console.log(`\n========================================`);
  console.log(`Upload complete in ${totalElapsed}s!`);
  console.log(`Model name: ${name}`);
  console.log(`Parts: ${parts.length}`);
  console.log(`Results:`);
  for (const r of results) {
    console.log(`  ${r.part}: ${r.result}`);
  }
  console.log(`========================================`);

  // Cleanup
  execSync(`rm -rf "${splitDir}"`);
}

main().catch((err) => {
  console.error('\nUpload failed:', err);
  process.exit(1);
});
