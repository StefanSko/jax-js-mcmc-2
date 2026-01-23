import { describe, it, expect } from 'vitest';
import type { IntegratorState, Integrator } from '../../src/integrators/types';
import { numpy as np } from '@jax-js/jax';

describe('Integrator types', () => {
  it('IntegratorState has required fields', () => {
    const state: IntegratorState = {
      position: np.array([0.0]),
      momentum: np.array([1.0]),
      logdensity: np.array(0.0),
      logdensityGrad: np.array([0.0]),
    };

    expect(state.position).toBeDefined();
    expect(state.momentum).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('Integrator type signature is correct', () => {
    const mockIntegrator: Integrator = (state, _stepSize) => state;
    expect(typeof mockIntegrator).toBe('function');
  });
});
