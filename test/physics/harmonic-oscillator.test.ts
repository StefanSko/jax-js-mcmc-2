import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@jax-js/jax';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Harmonic Oscillator', () => {
  const k = 1.0;
  const m = 1.0;

  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5 * k);
  };

  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5 / m);
  };

  it('trajectory matches analytical solution', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const q0 = 0.0;
    const p0 = 1.0;
    const stepSize = 0.01;
    const numSteps = 100; // t = 1.0

    const initialPosition = np.array(q0);
    const initialMomentum = np.array(p0);
    let state: IntegratorState = {
      position: initialPosition.ref,
      momentum: initialMomentum.ref,
      logdensity: logdensityFn(initialPosition.ref),
      logdensityGrad: grad(logdensityFn)(initialPosition),
    };
    initialMomentum.dispose();

    for (let i = 0; i < numSteps; i++) {
      state = integrator(state, stepSize);
    }

    const expectedQ = Math.sin(1.0);
    const expectedP = Math.cos(1.0);

    expect(state.position).toBeAllclose(expectedQ, { atol: 1e-4 });
    expect(state.momentum).toBeAllclose(expectedP, { atol: 1e-4 });

    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
