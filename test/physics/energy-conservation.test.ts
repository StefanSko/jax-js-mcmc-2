import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@jax-js/jax';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Energy Conservation', () => {
  const k = 1.0;
  const m = 1.0;

  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5 * k);
  };

  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5 / m);
  };

  const computeEnergy = (state: IntegratorState): number => {
    // H = -log π(q) + K(p) = U(q) + K(p)
    // Need refs since we're using these values without consuming the state
    const potential = state.logdensity.ref.neg();
    const kinetic = kineticEnergyFn(state.momentum.ref);
    return potential.add(kinetic).js() as number;
  };

  it('conserves energy over many steps', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const q0 = 1.0;
    const p0 = 0.5;
    const stepSize = 0.01;
    const numSteps = 1000;

    // Initial state
    const initialPosition = np.array(q0);
    const initialMomentum = np.array(p0);
    let state: IntegratorState = {
      position: initialPosition.ref,
      momentum: initialMomentum.ref,
      logdensity: logdensityFn(initialPosition.ref),
      logdensityGrad: grad(logdensityFn)(initialPosition),
    };
    initialMomentum.dispose();

    const initialEnergy = computeEnergy(state);

    // Integrate
    for (let i = 0; i < numSteps; i++) {
      state = integrator(state, stepSize);
    }

    const finalEnergy = computeEnergy(state);

    // Energy should be conserved to high precision
    expect(Math.abs(finalEnergy - initialEnergy)).toBeLessThan(1e-4);

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
