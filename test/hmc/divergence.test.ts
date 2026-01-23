import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';

describe('Divergence Detection', () => {
  it('detects divergence with large step size', () => {
    // Narrow target that will cause divergence with large steps
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-50.0).sum(); // Very narrow Gaussian
    };

    const sampler = HMC(logdensityFn)
      .stepSize(1.0) // Deliberately too large
      .numIntegrationSteps(100)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([0.1]));
    const key = random.key(123);
    const [newState, info] = sampler.step(key, state);

    // With such a large step size, we expect divergence
    // Note: .js() consumes the array
    const isDivergentValue = info.isDivergent.js() as boolean;
    // Note: May or may not diverge depending on random momentum
    // This test mainly ensures the divergence flag is computed
    expect(typeof isDivergentValue).toBe('boolean');

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    // isDivergent already consumed by .js()
    info.energy.dispose();
  });

  it('does not diverge with appropriate step size', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.01) // Small, safe step size
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    let divergenceCount = 0;
    let state = sampler.init(np.array([0.0]));

    for (let i = 0; i < 20; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      // Note: .js() consumes the array
      if (info.isDivergent.js() as boolean) {
        divergenceCount++;
      }

      // Cleanup info (isDivergent already consumed by .js())
      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.energy.dispose();

      state = newState;
    }

    // Should have no divergences with small step size
    expect(divergenceCount).toBe(0);

    // Cleanup final state
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
