import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';

describe('HMC Info', () => {
  it('returns diagnostics with correct shapes and types', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(5)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(7);
    const [newState, info] = sampler.step(key, state);

    expect(info.momentum.shape).toEqual([1]);
    expect(info.acceptanceProb.shape).toEqual([]);
    expect(info.isAccepted.shape).toEqual([]);
    expect(info.isDivergent.shape).toEqual([]);
    expect(info.energy.shape).toEqual([]);
    expect(info.numIntegrationSteps).toBe(5);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });
});
