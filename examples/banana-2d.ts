import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../src';

const b = 0.1;
const maskX = np.array([1.0, 0.0]);
const maskY = np.array([0.0, 1.0]);

const logdensityFn = (q: np.Array): np.Array => {
  const x = q.ref.mul(maskX.ref).sum();
  const y = q.ref.mul(maskY.ref).sum();

  const x2 = x.ref.mul(x);
  const yCentered = y.ref.sub(x2.ref.mul(b));

  const term1 = x2.mul(-0.5);
  const term2 = yCentered.ref.mul(yCentered).mul(-0.5);

  y.dispose();

  return term1.add(term2);
};

const sampler = HMC(logdensityFn)
  .stepSize(0.05)
  .numIntegrationSteps(1)
  .inverseMassMatrix(np.array([1.0, 1.0]))
  .valueAndGrad({ jit: true })
  .build();

let state = sampler.init(np.array([0.0, 0.0]));

const warmup = 300;
const numSamples = 800;

let sum0 = 0;
let sum1 = 0;

for (let i = 0; i < warmup + numSamples; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);

  if (i >= warmup) {
    const sampleJs = newState.position.ref.js() as number[] | number;
    const sampleArr = Array.isArray(sampleJs) ? sampleJs : [sampleJs];
    sum0 += sampleArr[0] ?? 0;
    sum1 += sampleArr[1] ?? 0;
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

console.log('Banana mean (approx):', [mean0.toFixed(3), mean1.toFixed(3)]);

state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();
maskX.dispose();
maskY.dispose();
