import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../src';

const logdensityFn = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

const sampler = HMC(logdensityFn)
  .stepSize(0.2)
  .numIntegrationSteps(3)
  .inverseMassMatrix(np.array([1.0]))
  .valueAndGrad({ jit: true })
  .build();

let state = sampler.init(np.array([0.0]));

const warmup = 200;
const numSamples = 500;
const samples: number[] = [];

for (let i = 0; i < warmup + numSamples; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);

  if (i >= warmup) {
    const sampleJs = newState.position.ref.js() as number[] | number;
    const sampleValue = Array.isArray(sampleJs) ? sampleJs[0] : sampleJs;
    samples.push(sampleValue);
  }

  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();

  state = newState;
}

const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

console.log('1D Gaussian mean:', mean.toFixed(3));
console.log('1D Gaussian variance:', variance.toFixed(3));

state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();
