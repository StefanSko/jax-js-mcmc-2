/**
 * Example: Sample from a 2D standard normal distribution
 *
 * Demonstrates HMC sampling in 2D with independent dimensions.
 *
 * Run with: npx tsx examples/sample-normal-2d.ts
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

/**
 * 2D Standard Gaussian distribution (independent dimensions)
 *
 * Target: N([0,0], I) - 2D standard normal
 * log p(x, y) = -0.5 * (x^2 + y^2)
 */

const logdensity = (q: np.Array): np.Array => {
  // q is [x, y]
  // log p = -0.5 * sum(q^2)
  return q.ref.mul(q).mul(-0.5).sum();
};

console.log('=== HMC Sampling from 2D Standard Normal ===\n');
console.log('Target: N([0,0], I) - independent 2D normal\n');

// Build sampler with 2D inverse mass matrix
// Using fewer integration steps to reduce memory pressure
const sampler = HMC(logdensity)
  .stepSize(0.2)
  .numIntegrationSteps(5)
  .inverseMassMatrix(np.array([1.0, 1.0]))
  .build();

// Initialize at origin
let state = sampler.init(np.array([0.0, 0.0]));
const samplesX: number[] = [];
const samplesY: number[] = [];
let acceptedCount = 0;
let divergentCount = 0;

// Reasonable iteration counts - grad() caching fix enables larger runs
const numWarmup = 200;
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

  // Record sample
  const posArray = newState.position.ref.js() as number[];
  if (posArray[0] !== undefined && posArray[1] !== undefined) {
    samplesX.push(posArray[0]);
    samplesY.push(posArray[1]);
  }

  // Track diagnostics - use .ref.js() to keep arrays alive for disposeInfo
  const accepted = info.isAccepted.ref.js() as boolean;
  const divergent = info.isDivergent.ref.js() as boolean;
  if (accepted) acceptedCount++;
  if (divergent) divergentCount++;

  disposeInfo(info);
  state = newState;
}

// Compute statistics
const meanX = samplesX.reduce((a, b) => a + b, 0) / samplesX.length;
const meanY = samplesY.reduce((a, b) => a + b, 0) / samplesY.length;
const varX = samplesX.reduce((a, b) => a + (b - meanX) ** 2, 0) / samplesX.length;
const varY = samplesY.reduce((a, b) => a + (b - meanY) ** 2, 0) / samplesY.length;

const acceptanceRate = acceptedCount / numSamples;
const divergenceRate = divergentCount / numSamples;

console.log('Results:');
console.log(`  Samples collected: ${samplesX.length}`);
console.log(`  Acceptance rate:   ${(acceptanceRate * 100).toFixed(1)}%`);
console.log(`  Divergence rate:   ${(divergenceRate * 100).toFixed(1)}%`);
console.log(`  Mean X:            ${meanX.toFixed(4)} (expected: ~0.0)`);
console.log(`  Mean Y:            ${meanY.toFixed(4)} (expected: ~0.0)`);
console.log(`  Var X:             ${varX.toFixed(4)}`);
console.log(`  Var Y:             ${varY.toFixed(4)}`);

// Show sample range
const minX = Math.min(...samplesX);
const maxX = Math.max(...samplesX);
const minY = Math.min(...samplesY);
const maxY = Math.max(...samplesY);

console.log('\nSample ranges:');
console.log(`  X: [${minX.toFixed(2)}, ${maxX.toFixed(2)}]`);
console.log(`  Y: [${minY.toFixed(2)}, ${maxY.toFixed(2)}]`);

// Show first few samples
console.log('\nFirst 5 samples (x, y):');
for (let i = 0; i < 5 && i < samplesX.length; i++) {
  console.log(`  (${samplesX[i]?.toFixed(3)}, ${samplesY[i]?.toFixed(3)})`);
}

// Cleanup
state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();

console.log('\nDone!');
