import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';
import type { HMCInfo } from '../../src/hmc/types';

describe('HMC Stationary Distribution', () => {
  const disposeInfo = (info: HMCInfo): void => {
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  };

  it('samples approximate standard normal mean and variance', () => {
    // Target: standard normal N(0, 1)
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

    // Warmup (fewer iterations for speed)
    for (let i = 0; i < 50; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);
      disposeInfo(info);
      state = newState;
    }

    // Sample
    for (let i = 50; i < 350; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      // Get the position value - use .ref to keep the array alive after .js()
      // .js() returns [value] for shape [1] arrays, so extract first element
      const posRef = newState.position.ref;
      const posArray = posRef.js() as number[];
      const posValue = posArray[0];
      if (posValue !== undefined) {
        samples.push(posValue);
      }

      disposeInfo(info);
      state = newState;
    }

    // Compute sample statistics
    const validSamples = samples.filter(s => Number.isFinite(s));
    expect(validSamples.length).toBeGreaterThan(0);

    const mean = validSamples.reduce((a, b) => a + b, 0) / validSamples.length;
    const variance = validSamples.reduce((a, b) => a + (b - mean) ** 2, 0) / validSamples.length;

    // Check approximate agreement with N(0, 1)
    expect(Math.abs(mean)).toBeLessThan(0.3); // Mean should be close to 0
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.5); // Variance should be close to 1

    // Cleanup
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
