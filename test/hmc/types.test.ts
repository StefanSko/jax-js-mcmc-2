import { describe, it, expect } from 'vitest';
import type { HMCState, HMCInfo, HMCConfig } from '../../src/hmc/types';
import '../../src/hmc/types';
import { numpy as np } from '@jax-js/jax';
import type { Integrator } from '../../src/integrators/types';

describe('HMC types', () => {
  it('HMCState has required fields', () => {
    const state: HMCState = {
      position: np.array([0.0]),
      logdensity: np.array(0.0),
      logdensityGrad: np.array([0.0]),
    };

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('HMCInfo has required fields', () => {
    const info: HMCInfo = {
      momentum: np.array([1.0]),
      acceptanceRate: 0.5,
      isAccepted: true,
      isDivergent: false,
      energy: 1.0,
      numIntegrationSteps: 10,
    };

    expect(info.momentum).toBeDefined();
    expect(info.acceptanceRate).toBeTypeOf('number');
    expect(info.isAccepted).toBeTypeOf('boolean');
    expect(info.isDivergent).toBeTypeOf('boolean');
    expect(info.energy).toBeTypeOf('number');
    expect(info.numIntegrationSteps).toBeTypeOf('number');

    info.momentum.dispose();
  });

  it('HMCConfig has required fields', () => {
    const integrator: Integrator = (state, _stepSize) => state;
    const config: HMCConfig = {
      stepSize: 0.1,
      numIntegrationSteps: 10,
      inverseMassMatrix: np.array([1.0]),
      divergenceThreshold: 1000,
      integrator,
    };

    expect(config.stepSize).toBeTypeOf('number');
    expect(config.numIntegrationSteps).toBeTypeOf('number');
    expect(config.inverseMassMatrix).toBeDefined();
    expect(config.divergenceThreshold).toBeTypeOf('number');
    expect(config.integrator).toBeTypeOf('function');

    config.inverseMassMatrix.dispose();
  });
});
