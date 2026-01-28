import { describe, it, expect } from 'vitest';
import { numpy as np, random, type Array } from '@jax-js/jax';
import { createRWMKernel } from '../../src/rwm/kernel';
import type { RWMState, RWMConfig } from '../../src/rwm/types';

describe('RWM Kernel', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  const createState = (values: number[]): RWMState => {
    const position = np.array(values);
    const positionRef = position.ref;
    position.dispose();
    return {
      position: positionRef,
      logdensity: logdensityFn(positionRef.ref),
    };
  };

  it('returns new state and info', () => {
    const config: RWMConfig = { logdensityFn, stepSize: 0.1 };
    const rwmStep = createRWMKernel(config);

    const state = createState([0.0]);
    const key = random.key(42);

    const [newState, info] = rwmStep(key, state);

    expect(newState.position).toBeDefined();
    expect(newState.logdensity).toBeDefined();

    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.proposedPosition).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });

  it('proposes q + stepSize * noise', () => {
    const stepSize = 0.2;
    const config: RWMConfig = { logdensityFn, stepSize };
    const rwmStep = createRWMKernel(config);

    const state = createState([1.0, -1.0]);
    const key = random.key(5);

    const [noiseKey] = random.split(key.ref, 2) as unknown as [Array, Array];
    const noise = random.normal(noiseKey, state.position.shape);
    const expectedProposal = state.position.ref.add(noise.mul(stepSize));

    const [newState, info] = rwmStep(key, state);

    expect(info.proposedPosition).toBeAllclose(expectedProposal, { atol: 1e-6 });

    expectedProposal.dispose();
    newState.position.dispose();
    newState.logdensity.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });

  it('accepts identical proposal when stepSize is zero', () => {
    const config: RWMConfig = { logdensityFn, stepSize: 0 };
    const rwmStep = createRWMKernel(config);

    const state = createState([0.5]);
    const key = random.key(9);

    const [newState, info] = rwmStep(key, state);
    const acceptanceProb = info.acceptanceProb.ref.js() as number;
    const isAccepted = info.isAccepted.ref.js() as boolean;

    expect(acceptanceProb).toBeCloseTo(1, 6);
    expect(isAccepted).toBe(true);

    newState.position.dispose();
    newState.logdensity.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });
});
