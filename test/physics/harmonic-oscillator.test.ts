import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@jax-js/jax';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Harmonic Oscillator', () => {
  // Potential: U(q) = 0.5 * k * q^2
  // Kinetic:  K(p) = 0.5 * p^2 / m
  // With k=1, m=1: ω = 1
  // Solution: q(t) = q0*cos(t) + p0*sin(t)
  //           p(t) = -q0*sin(t) + p0*cos(t)

  const k = 1.0;
  const m = 1.0;

  // log π(q) = -U(q) = -0.5 * q^2
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5 * k);
  };

  // K(p) = 0.5 * p^2 / m
  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5 / m);
  };

  it('trajectory matches analytical solution', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const q0 = 0.0;
    const p0 = 1.0;
    const stepSize = 0.01;
    const numSteps = 100; // t = 1.0

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

    // Integrate
    for (let i = 0; i < numSteps; i++) {
      state = integrator(state, stepSize);
    }

    // Analytical solution at t=1: q(1) = sin(1), p(1) = cos(1)
    const expectedQ = Math.sin(1.0);
    const expectedP = Math.cos(1.0);

    expect(state.position).toBeAllclose(expectedQ, { atol: 1e-4 });
    expect(state.momentum).toBeAllclose(expectedP, { atol: 1e-4 });

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
