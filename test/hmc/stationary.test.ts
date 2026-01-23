import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';

describe('HMC Stationary Distribution', () => {
  it('samples approximate standard normal mean and variance', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    let state = sampler.init(np.array([0.0]));
    const samples: number[] = [];

    for (let i = 0; i < 50; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);
      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();
      state = newState;
    }

    for (let i = 50; i < 250; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      const sampleJs = newState.position.ref.js() as number[] | number;
      let sampleValue: number;
      if (Array.isArray(sampleJs)) {
        const first = sampleJs[0];
        if (first === undefined) {
          throw new Error('Expected scalar sample');
        }
        sampleValue = first;
      } else {
        sampleValue = sampleJs;
      }
      samples.push(sampleValue);

      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();
      state = newState;
    }

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    expect(Math.abs(mean)).toBeLessThan(0.2);
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.3);

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
