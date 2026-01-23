/**
 * Example: Sample from a standard normal distribution N(0, 1)
 *
 * Run with: npx tsx examples/sample-normal.ts
 */

import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../src';
import type { HMCInfo } from '../src';

// Helper to dispose HMCInfo arrays
function disposeInfo(info: HMCInfo): void {
  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();
}

// Target: standard normal N(0, 1)
// log p(x) = -0.5 * x^2 (up to constant)
const logdensity = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

console.log('=== HMC Sampling from Standard Normal ===\n');

// Build sampler
const sampler = HMC(logdensity)
  .stepSize(0.1)
  .numIntegrationSteps(10)
  .inverseMassMatrix(np.array([1.0]))
  .build();

// Initialize
let state = sampler.init(np.array([0.0]));
const samples: number[] = [];
let acceptedCount = 0;

const numWarmup = 100;
const numSamples = 500;

console.log(`Warmup: ${numWarmup} iterations`);
console.log(`Sampling: ${numSamples} iterations\n`);

// Warmup
for (let i = 0; i < numWarmup; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);
  disposeInfo(info);
  state = newState;
}

// Sample
for (let i = 0; i < numSamples; i++) {
  const key = random.key(numWarmup + i);
  const [newState, info] = sampler.step(key, state);

  // Record sample (use .ref to keep position alive after .js())
  const posArray = newState.position.ref.js() as number[];
  const value = posArray[0];
  if (value !== undefined) {
    samples.push(value);
  }

  // Track acceptance - use .ref.js() to keep array alive for disposeInfo
  const accepted = info.isAccepted.ref.js() as boolean;
  if (accepted) acceptedCount++;

  disposeInfo(info);
  state = newState;
}

// Compute statistics
const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;
const std = Math.sqrt(variance);
const acceptanceRate = acceptedCount / numSamples;

console.log('Results:');
console.log(`  Samples collected: ${samples.length}`);
console.log(`  Acceptance rate:   ${(acceptanceRate * 100).toFixed(1)}%`);
console.log(`  Sample mean:       ${mean.toFixed(4)} (expected: 0.0)`);
console.log(`  Sample std:        ${std.toFixed(4)} (expected: 1.0)`);
console.log(`  Sample variance:   ${variance.toFixed(4)} (expected: 1.0)`);

// Show some samples
console.log('\nFirst 10 samples:');
console.log(`  [${samples.slice(0, 10).map(s => s.toFixed(3)).join(', ')}]`);

// Cleanup
state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();

console.log('\nDone!');
