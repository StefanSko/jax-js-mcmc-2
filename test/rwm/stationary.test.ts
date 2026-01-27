import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { RWM } from '../../src/rwm/builder';

describe('RWM Stationary Distribution', () => {
  it('samples approximate standard normal mean and variance', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = RWM(logdensityFn).stepSize(0.5).build();

    let state = sampler.init(np.array([0.0]));
    const samples: number[] = [];

    for (let i = 0; i < 200; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.proposedPosition.dispose();
      state = newState;
    }

    for (let i = 200; i < 1000; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      const sampleJs = newState.position.ref.js() as number[] | number;
      const sampleValue = Array.isArray(sampleJs) ? sampleJs[0] ?? 0 : sampleJs;
      samples.push(sampleValue);

      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.proposedPosition.dispose();
      state = newState;
    }

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    expect(Math.abs(mean)).toBeLessThan(0.35);
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.4);

    state.position.dispose();
    state.logdensity.dispose();
  });
});
