import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@jax-js/jax';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Gradient Flow', () => {
  it('grad flows through single integrator step', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5);
    };

    const kineticEnergyFn = (p: np.Array): np.Array => {
      return p.ref.mul(p).mul(0.5);
    };

    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    // Differentiate final position w.r.t. initial position
    const finalPosition = (q0: np.Array): np.Array => {
      const position = q0.ref;
      const momentum = np.array(1.0);
      let state: IntegratorState = {
        position: position,
        momentum: momentum.ref,
        logdensity: logdensityFn(position.ref),
        logdensityGrad: grad(logdensityFn)(position.ref),
      };
      momentum.dispose();

      state = integrator(state, 0.01);

      // Dispose unused arrays
      state.momentum.dispose();
      state.logdensity.dispose();
      state.logdensityGrad.dispose();

      return state.position;
    };

    const dqdq0 = grad(finalPosition)(np.array(0.0));

    // Gradient should be close to 1 for small step size
    expect(dqdq0).toBeAllclose(1.0, { atol: 0.01 });
  });

  it('grad through multiple integrator steps', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5);
    };

    const kineticEnergyFn = (p: np.Array): np.Array => {
      return p.ref.mul(p).mul(0.5);
    };

    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const finalPosition = (q0: np.Array): np.Array => {
      const position = q0.ref;
      const momentum = np.array(1.0);
      let state: IntegratorState = {
        position: position,
        momentum: momentum.ref,
        logdensity: logdensityFn(position.ref),
        logdensityGrad: grad(logdensityFn)(position.ref),
      };
      momentum.dispose();

      for (let i = 0; i < 10; i++) {
        state = integrator(state, 0.01);
      }

      state.momentum.dispose();
      state.logdensity.dispose();
      state.logdensityGrad.dispose();

      return state.position;
    };

    const gradient = grad(finalPosition)(np.array(0.5));

    // Should return a valid scalar gradient
    expect(gradient.shape).toEqual([]);
    // Note: .js() consumes the array via dataSync, so no dispose needed
    const gradValue = gradient.js() as number;
    expect(typeof gradValue).toBe('number');
    expect(Number.isFinite(gradValue)).toBe(true);
  });
});
