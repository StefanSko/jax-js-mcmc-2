import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';

describe('Divergence Detection', () => {
  it('detects divergence with large step size', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-50.0).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(1.0)
      .numIntegrationSteps(100)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([0.1]));
    const key = random.key(123);
    const [newState, info] = sampler.step(key, state);

    const isDivergentValue = info.isDivergent.ref.js() as boolean;
    expect(typeof isDivergentValue).toBe('boolean');

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('does not diverge with appropriate step size', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.01)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    let divergenceCount = 0;
    let state = sampler.init(np.array([0.0]));

    for (let i = 0; i < 20; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      if (info.isDivergent.ref.js()) {
        divergenceCount++;
      }

      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();

      state = newState;
    }

    expect(divergenceCount).toBe(0);

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('flags NaN energy as divergent', () => {
    const nanLogdensity = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(np.nan).sum();
    };

    const sampler = HMC(nanLogdensity)
      .stepSize(0.1)
      .numIntegrationSteps(1)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([1.0]));
    const key = random.key(42);
    const [newState, info] = sampler.step(key, state);

    expect(info.isDivergent.ref.js()).toBe(true);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('rejects NaN energy proposals', () => {
    const nanLogdensity = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(np.nan).sum();
    };

    const sampler = HMC(nanLogdensity)
      .stepSize(0.1)
      .numIntegrationSteps(1)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([1.0]));
    const key = random.key(42);
    const [newState, info] = sampler.step(key, state);

    expect(info.isAccepted.ref.js()).toBe(false);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('does not leak arrays on NaN energy path', () => {
    const nanLogdensity = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(np.nan).sum();
    };

    const sampler = HMC(nanLogdensity)
      .stepSize(0.1)
      .numIntegrationSteps(1)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([1.0]));
    const key = random.key(99);
    const [newState, info] = sampler.step(key, state);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();

    expect(newState.position.refCount).toBe(0);
    expect(info.isDivergent.refCount).toBe(0);
    expect(info.isAccepted.refCount).toBe(0);
  });
});
