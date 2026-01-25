import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../src';

// Test: Does calling .sync() periodically prevent memory growth?

const logdensityFn = (q: np.Array): np.Array => q.ref.mul(q).mul(-0.5).sum();

const iterations = Number.parseInt(process.env.ITERATIONS ?? '150', 10);
const logEvery = Number.parseInt(process.env.LOG_EVERY ?? '30', 10);
const syncEvery = Number.parseInt(process.env.SYNC_EVERY ?? '100', 10);
const stepSize = Number.parseFloat(process.env.STEP_SIZE ?? '0.2');
const numIntegrationSteps = Number.parseInt(
  process.env.INTEGRATION_STEPS ?? '3',
  10
);

const sampler = HMC(logdensityFn)
  .stepSize(stepSize)
  .numIntegrationSteps(numIntegrationSteps)
  .inverseMassMatrix(np.array([1.0]))
  .jitStep()
  .build();

let state = sampler.init(np.array([0.0]));

const logMemory = (label: string): void => {
  if (globalThis.gc) {
    globalThis.gc();
  }
  const { heapUsed, rss } = process.memoryUsage();
  const heapMb = (heapUsed / 1024 / 1024).toFixed(1);
  const rssMb = (rss / 1024 / 1024).toFixed(1);
  console.log(`${label} heap=${heapMb}MB rss=${rssMb}MB`);
};

logMemory('start');

for (let i = 0; i < iterations; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);

  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();

  state = newState;

  // Periodically read values to drain pending queue (forces kernel submission)
  // Use .ref before .js() to prevent dispose, then dispose the ref'd copy
  if ((i + 1) % syncEvery === 0) {
    const posRef = state.position.ref;
    posRef.js();  // Forces pending kernels to execute (and disposes posRef)
  }

  if ((i + 1) % logEvery === 0) {
    logMemory(`iter ${i + 1}`);
  }
}

logMemory('end');

state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();
