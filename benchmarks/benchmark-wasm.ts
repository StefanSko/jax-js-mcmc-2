#!/usr/bin/env npx tsx
/**
 * WASM Backend Benchmark
 *
 * Run with:
 *   NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" npx tsx benchmarks/benchmark-wasm.ts
 *
 * Options:
 *   OUTPUT=results.json    Write results to JSON file
 *   CONFIGS=default|scaling|all   Choose benchmark configs
 */

import {
  DEFAULT_CONFIGS,
  SCALING_CONFIGS,
  initBackend,
  runBenchmark,
  type BenchmarkResult,
} from './benchmark-runner';

async function main() {
  const outputFile = process.env.OUTPUT ?? 'benchmarks/results-wasm.json';
  const configSet = process.env.CONFIGS ?? 'default';

  console.log('='.repeat(60));
  console.log('JAX-JS MCMC WASM Benchmark');
  console.log('='.repeat(60));

  // Initialize backend
  console.log('\nInitializing WASM backend...');
  try {
    const devices = await initBackend('wasm');
    console.log(`Available devices: ${devices.join(', ')}`);
    console.log('Using: wasm');
  } catch (e) {
    console.error('Failed to initialize WASM backend:', e);
    process.exit(1);
  }

  // Select configs
  let configs = DEFAULT_CONFIGS;
  if (configSet === 'scaling') {
    configs = SCALING_CONFIGS;
  } else if (configSet === 'all') {
    configs = [...DEFAULT_CONFIGS, ...SCALING_CONFIGS];
  }

  console.log(`\nRunning ${configs.length} benchmark configurations...\n`);

  const results: BenchmarkResult[] = [];

  for (const config of configs) {
    console.log(`Running: ${config.name}`);
    console.log(
      `  Sampler: ${config.sampler}, Dims: ${config.dimensions}, ` +
        `Iters: ${config.iterations}, Warmup: ${config.warmup}`
    );

    try {
      const result = await runBenchmark(config, 'wasm', true);
      results.push(result);

      console.log(`  Time: ${result.totalTime.toFixed(1)}ms`);
      console.log(`  Throughput: ${result.iterationsPerSecond.toFixed(1)} iter/s`);
      console.log(`  Avg time/iter: ${result.avgTimePerIteration.toFixed(3)}ms`);
      if (result.memoryUsedMB > 0) {
        console.log(`  Memory: ${result.memoryUsedMB.toFixed(1)}MB`);
      }
      console.log();
    } catch (e) {
      console.error(`  ERROR: ${e}`);
      console.log();
    }
  }

  // Write results to file
  const fs = await import('fs');
  const output = {
    backend: 'wasm',
    timestamp: new Date().toISOString(),
    platform: process.platform,
    nodeVersion: process.version,
    results,
  };

  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));
  console.log(`\nResults written to: ${outputFile}`);

  // Print summary table
  console.log('\n' + '='.repeat(60));
  console.log('Summary');
  console.log('='.repeat(60));
  console.log(
    'Name'.padEnd(25) +
      'Dims'.padStart(6) +
      'Iters/s'.padStart(12) +
      'ms/iter'.padStart(12)
  );
  console.log('-'.repeat(55));

  for (const r of results) {
    console.log(
      r.config.name.padEnd(25) +
        String(r.config.dimensions).padStart(6) +
        r.iterationsPerSecond.toFixed(1).padStart(12) +
        r.avgTimePerIteration.toFixed(3).padStart(12)
    );
  }
}

main().catch(console.error);
