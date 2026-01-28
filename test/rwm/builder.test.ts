import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { RWM } from '../../src/rwm/builder';

describe('RWM Builder', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  it('builds sampler with fluent API', () => {
    const sampler = RWM(logdensityFn).stepSize(0.1).build();

    expect(sampler.init).toBeDefined();
    expect(sampler.step).toBeDefined();
  });

  it('builder is immutable', () => {
    const builder1 = RWM(logdensityFn);
    const builder2 = builder1.stepSize(0.1);

    expect(builder1).not.toBe(builder2);
  });

  it('init creates valid state', () => {
    const sampler = RWM(logdensityFn).stepSize(0.1).build();

    const position = np.array([1.0]);
    const state = sampler.init(position);

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
  });

  it('step runs without error', () => {
    const sampler = RWM(logdensityFn).stepSize(0.1).build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(42);
    const [newState, info] = sampler.step(key, state);

    expect(newState.position).toBeDefined();
    expect(info.isAccepted).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });

  it('jitStep path runs without error', () => {
    const sampler = RWM(logdensityFn).stepSize(0.1).jitStep().build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(13);
    const [newState, info] = sampler.step(key, state);

    expect(newState.position).toBeDefined();
    expect(info.acceptanceProb).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });

  it('throws if required config is missing', () => {
    expect(() => RWM(logdensityFn).build()).toThrow('stepSize required');
  });
});
