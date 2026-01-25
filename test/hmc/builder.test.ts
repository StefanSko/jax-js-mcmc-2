import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';

describe('HMC Builder', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  it('builds sampler with fluent API', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    expect(sampler.init).toBeDefined();
    expect(sampler.step).toBeDefined();
  });

  it('builder is immutable', () => {
    const builder1 = HMC(logdensityFn);
    const builder2 = builder1.stepSize(0.1);

    expect(builder1).not.toBe(builder2);
  });

  it('init creates valid state', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    const position = np.array([1.0]);
    const state = sampler.init(position);

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('step runs without error', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(42);
    const [newState, info] = sampler.step(key, state);

    expect(newState.position).toBeDefined();
    expect(info.isAccepted).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('does not expose valueAndGrad (API kept minimal)', () => {
    const builder = HMC(logdensityFn);
    expect('valueAndGrad' in builder).toBe(false);
  });

  it('jitStep path runs without error', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(5)
      .inverseMassMatrix(np.array([1.0]))
      .jitStep()
      .build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(13);
    const [newState, info] = sampler.step(key, state);

    expect(info.numIntegrationSteps).toBe(5);
    expect(newState.position).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('throws if required config is missing', () => {
    expect(() => HMC(logdensityFn).build()).toThrow('stepSize required');
    expect(() => HMC(logdensityFn).stepSize(0.1).build()).toThrow('numIntegrationSteps required');
  });
});
