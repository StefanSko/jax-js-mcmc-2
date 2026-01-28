/**
 * JAX-JS MCMC Benchmark Runner
 *
 * Core benchmark logic for measuring HMC and RWM sampler performance
 * across different backends (WASM, WebGPU).
 */

import { numpy as np, random, init, defaultDevice } from '@jax-js/jax';
import { HMC } from '../src/hmc';
import { RWM } from '../src/rwm';

// Types for benchmark results
export interface BenchmarkConfig {
  name: string;
  sampler: 'hmc' | 'rwm';
  dimensions: number;
  iterations: number;
  warmup: number;
  distribution: 'gaussian' | 'banana' | 'correlated';
  stepSize?: number;
  numIntegrationSteps?: number;
}

export interface BenchmarkResult {
  config: BenchmarkConfig;
  backend: string;
  totalTime: number;
  iterationsPerSecond: number;
  avgTimePerIteration: number;
  memoryUsedMB: number;
  timestamp: string;
}

// Log density functions for different distributions
export function gaussianLogDensity(dimensions: number) {
  return (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };
}

export function bananaLogDensity(b = 0.1) {
  const maskX = np.array([1.0, 0.0]);
  const maskY = np.array([0.0, 1.0]);

  return (q: np.Array): np.Array => {
    const x = q.ref.mul(maskX.ref).sum();
    const y = q.ref.mul(maskY.ref).sum();
    const x2 = x.ref.mul(x);
    const yCentered = y.ref.sub(x2.ref.mul(b));
    const term1 = x2.mul(-0.5);
    const term2 = yCentered.ref.mul(yCentered).mul(-0.5);
    y.dispose();
    return term1.add(term2);
  };
}

export function correlatedGaussianLogDensity(dimensions: number, rho = 0.9) {
  // Correlated Gaussian with covariance matrix having off-diagonal elements
  return (q: np.Array): np.Array => {
    // Simple approximation: scale by inverse covariance
    const factor = 1 / (1 - rho * rho);
    return q.ref.mul(q).mul(-0.5 * factor).sum();
  };
}

// Create initial position for given dimensions
export function createInitialPosition(dimensions: number): np.Array {
  return np.zeros([dimensions]);
}

// Create inverse mass matrix for given dimensions
export function createInverseMassMatrix(dimensions: number): np.Array {
  return np.ones([dimensions]);
}

// Build sampler based on config
export function buildSampler(config: BenchmarkConfig, useJit = true) {
  const { sampler, dimensions, distribution, stepSize, numIntegrationSteps } =
    config;

  let logdensityFn: (q: np.Array) => np.Array;

  switch (distribution) {
    case 'gaussian':
      logdensityFn = gaussianLogDensity(dimensions);
      break;
    case 'banana':
      if (dimensions !== 2) {
        throw new Error('Banana distribution only supports 2 dimensions');
      }
      logdensityFn = bananaLogDensity();
      break;
    case 'correlated':
      logdensityFn = correlatedGaussianLogDensity(dimensions);
      break;
    default:
      throw new Error(`Unknown distribution: ${distribution}`);
  }

  if (sampler === 'hmc') {
    const builder = HMC(logdensityFn)
      .stepSize(stepSize ?? 0.1)
      .numIntegrationSteps(numIntegrationSteps ?? 10)
      .inverseMassMatrix(createInverseMassMatrix(dimensions));

    return useJit ? builder.jitStep().build() : builder.build();
  } else {
    // RWM doesn't have inverseMassMatrix in builder API
    const builder = RWM(logdensityFn).stepSize(stepSize ?? 0.5);

    return useJit ? builder.jitStep().build() : builder.build();
  }
}

// Run a single benchmark
export async function runBenchmark(
  config: BenchmarkConfig,
  backend: string,
  useJit = true
): Promise<BenchmarkResult> {
  const { dimensions, iterations, warmup } = config;

  // Build sampler
  const samplerObj = buildSampler(config, useJit);
  let state = samplerObj.init(createInitialPosition(dimensions));

  // Helper to dispose info arrays (handles both HMC and RWM)
  const disposeInfo = (info: Record<string, np.Array | undefined>): void => {
    for (const key of Object.keys(info)) {
      const val = info[key];
      if (val && typeof val.dispose === 'function') {
        val.dispose();
      }
    }
  };

  // Warmup phase (not timed)
  for (let i = 0; i < warmup; i++) {
    const key = random.key(i);
    const [newState, info] = samplerObj.step(key, state);
    disposeInfo(info as unknown as Record<string, np.Array>);
    state = newState;
  }

  // Force sync and GC before timed section
  if (typeof globalThis.gc === 'function') {
    globalThis.gc();
  }

  // Timed section
  const startTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    const key = random.key(warmup + i);
    const [newState, info] = samplerObj.step(key, state);
    disposeInfo(info as unknown as Record<string, np.Array>);
    state = newState;
  }

  const endTime = performance.now();
  const totalTime = endTime - startTime;

  // Get memory usage
  let memoryUsedMB = 0;
  if (typeof process !== 'undefined' && process.memoryUsage) {
    if (typeof globalThis.gc === 'function') {
      globalThis.gc();
    }
    memoryUsedMB = process.memoryUsage().heapUsed / 1024 / 1024;
  }

  // Cleanup state
  state.position.dispose();
  state.logdensity.dispose();
  if ('logdensityGrad' in state) {
    (state as { logdensityGrad: np.Array }).logdensityGrad.dispose();
  }

  return {
    config,
    backend,
    totalTime,
    iterationsPerSecond: (iterations / totalTime) * 1000,
    avgTimePerIteration: totalTime / iterations,
    memoryUsedMB,
    timestamp: new Date().toISOString(),
  };
}

// Default benchmark configurations
export const DEFAULT_CONFIGS: BenchmarkConfig[] = [
  // HMC benchmarks - varying dimensions
  {
    name: 'HMC-1D-Gaussian',
    sampler: 'hmc',
    dimensions: 1,
    iterations: 500,
    warmup: 50,
    distribution: 'gaussian',
    stepSize: 0.2,
    numIntegrationSteps: 10,
  },
  {
    name: 'HMC-10D-Gaussian',
    sampler: 'hmc',
    dimensions: 10,
    iterations: 500,
    warmup: 50,
    distribution: 'gaussian',
    stepSize: 0.1,
    numIntegrationSteps: 10,
  },
  {
    name: 'HMC-50D-Gaussian',
    sampler: 'hmc',
    dimensions: 50,
    iterations: 300,
    warmup: 30,
    distribution: 'gaussian',
    stepSize: 0.05,
    numIntegrationSteps: 20,
  },
  {
    name: 'HMC-100D-Gaussian',
    sampler: 'hmc',
    dimensions: 100,
    iterations: 200,
    warmup: 20,
    distribution: 'gaussian',
    stepSize: 0.03,
    numIntegrationSteps: 30,
  },

  // RWM benchmarks for comparison
  {
    name: 'RWM-1D-Gaussian',
    sampler: 'rwm',
    dimensions: 1,
    iterations: 500,
    warmup: 50,
    distribution: 'gaussian',
    stepSize: 2.0,
  },
  {
    name: 'RWM-10D-Gaussian',
    sampler: 'rwm',
    dimensions: 10,
    iterations: 500,
    warmup: 50,
    distribution: 'gaussian',
    stepSize: 0.5,
  },
  {
    name: 'RWM-50D-Gaussian',
    sampler: 'rwm',
    dimensions: 50,
    iterations: 300,
    warmup: 30,
    distribution: 'gaussian',
    stepSize: 0.2,
  },
];

// Scaling benchmark for GPU benefit analysis
export const SCALING_CONFIGS: BenchmarkConfig[] = [
  ...[1, 5, 10, 20, 50, 100, 200].map(
    (dim): BenchmarkConfig => ({
      name: `HMC-${dim}D-Scaling`,
      sampler: 'hmc',
      dimensions: dim,
      iterations: 500,
      warmup: 50,
      distribution: 'gaussian',
      stepSize: Math.max(0.01, 0.2 / Math.sqrt(dim)),
      numIntegrationSteps: Math.min(50, 10 + dim / 5),
    })
  ),
];

// Initialize JAX-JS with specified backend
export async function initBackend(
  backend: 'wasm' | 'webgpu'
): Promise<string[]> {
  const devices = await init();
  if (devices.includes(backend)) {
    defaultDevice(backend);
    return devices;
  }
  throw new Error(`Backend ${backend} not available. Available: ${devices}`);
}
