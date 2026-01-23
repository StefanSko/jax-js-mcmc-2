import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../src';

const logdensityFn = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

const sampler = HMC(logdensityFn)
  .stepSize(0.2)
  .numIntegrationSteps(3)
  .inverseMassMatrix(np.array([1.0, 1.0]))
  .valueAndGrad({ jit: true })
  .build();

let state = sampler.init(np.array([0.0, 0.0]));

const warmup = 200;
const numSamples = 500;

let sum0 = 0;
let sum1 = 0;
let sumsq0 = 0;
let sumsq1 = 0;

for (let i = 0; i < warmup + numSamples; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);

  if (i >= warmup) {
    const sampleJs = newState.position.ref.js() as number[] | number;
    const sampleArr = Array.isArray(sampleJs) ? sampleJs : [sampleJs];
    const x = sampleArr[0] ?? 0;
    const y = sampleArr[1] ?? 0;
    sum0 += x;
    sum1 += y;
    sumsq0 += x * x;
    sumsq1 += y * y;
  }

  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();

  state = newState;
}

const mean0 = sum0 / numSamples;
const mean1 = sum1 / numSamples;
const var0 = sumsq0 / numSamples - mean0 * mean0;
const var1 = sumsq1 / numSamples - mean1 * mean1;

console.log('2D Gaussian mean:', [mean0.toFixed(3), mean1.toFixed(3)]);
console.log('2D Gaussian variance:', [var0.toFixed(3), var1.toFixed(3)]);

state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();
