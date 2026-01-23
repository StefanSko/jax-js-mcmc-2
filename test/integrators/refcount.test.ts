import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@jax-js/jax';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Integrator Memory Management', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5);
  };

  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5);
  };

  const createInitialState = (): IntegratorState => {
    const position = np.array(1.0);
    const momentum = np.array(0.5);
    const positionRef = position.ref;
    const momentumRef = momentum.ref;
    position.dispose();
    momentum.dispose();
    return {
      position: positionRef,
      momentum: momentumRef,
      logdensity: logdensityFn(positionRef.ref),
      logdensityGrad: grad(logdensityFn)(positionRef.ref),
    };
  };

  it('old state arrays have refCount 0 after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const oldPosition = state.position;
    const oldMomentum = state.momentum;
    const oldLogdensity = state.logdensity;
    const oldLogdensityGrad = state.logdensityGrad;

    const newState = integrator(state, 0.01);

    expect(oldPosition.refCount).toBe(0);
    expect(oldMomentum.refCount).toBe(0);
    expect(oldLogdensity.refCount).toBe(0);
    expect(oldLogdensityGrad.refCount).toBe(0);

    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('new state arrays have refCount 1 after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const newState = integrator(state, 0.01);

    expect(newState.position.refCount).toBe(1);
    expect(newState.momentum.refCount).toBe(1);
    expect(newState.logdensity.refCount).toBe(1);
    expect(newState.logdensityGrad.refCount).toBe(1);

    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('old state arrays throw ReferenceError when accessed after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const oldPosition = state.position;
    const newState = integrator(state, 0.01);

    expect(() => {
      oldPosition.js();
    }).toThrowError(ReferenceError);

    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('no reference accumulation over many iterations', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    let state = createInitialState();

    for (let i = 0; i < 100; i++) {
      const oldPosition = state.position;
      state = integrator(state, 0.01);

      expect(oldPosition.refCount).toBe(0);
      expect(state.position.refCount).toBe(1);
    }

    expect(state.position.refCount).toBe(1);

    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
