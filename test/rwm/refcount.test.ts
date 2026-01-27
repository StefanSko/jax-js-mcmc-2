import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { createRWMKernel } from '../../src/rwm/kernel';
import type { RWMConfig, RWMState, RWMInfo } from '../../src/rwm/types';

const logdensityFn = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

const createConfig = (): RWMConfig => ({
  logdensityFn,
  stepSize: 0.1,
});

const createInitialState = (): RWMState => {
  const position = np.array([0.0]);
  const positionRef = position.ref;
  position.dispose();
  return {
    position: positionRef,
    logdensity: logdensityFn(positionRef.ref),
  };
};

const disposeInfo = (info: RWMInfo): void => {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.proposedPosition.dispose();
};

describe('RWM kernel refcount', () => {
  it('old state arrays have refCount 0 after step', () => {
    const config = createConfig();
    const rwmStep = createRWMKernel(config);

    const state = createInitialState();
    const oldPosition = state.position;
    const oldLogdensity = state.logdensity;

    const key = random.key(42);
    const [newState, info] = rwmStep(key, state);

    expect(oldPosition.refCount).toBe(0);
    expect(oldLogdensity.refCount).toBe(0);

    newState.position.dispose();
    newState.logdensity.dispose();
    disposeInfo(info);
  });

  it('new state arrays have refCount 1 after step', () => {
    const config = createConfig();
    const rwmStep = createRWMKernel(config);

    const state = createInitialState();
    const key = random.key(42);
    const [newState, info] = rwmStep(key, state);

    expect(newState.position.refCount).toBe(1);
    expect(newState.logdensity.refCount).toBe(1);

    newState.position.dispose();
    newState.logdensity.dispose();
    disposeInfo(info);
  });

  it('no reference accumulation over many RWM steps', () => {
    const config = createConfig();
    const rwmStep = createRWMKernel(config);

    let state = createInitialState();

    for (let i = 0; i < 50; i++) {
      const oldPosition = state.position;
      const key = random.key(i);
      const [newState, info] = rwmStep(key, state);

      expect(oldPosition.refCount).toBe(0);
      expect(newState.position.refCount).toBe(1);

      disposeInfo(info);
      state = newState;
    }

    expect(state.position.refCount).toBe(1);

    state.position.dispose();
    state.logdensity.dispose();
  });
});
