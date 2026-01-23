import { describe, it, expect } from 'vitest';
import type { HMCState, HMCInfo, HMCConfig } from '../../src/hmc/types';
import '../../src/hmc/types';
import { numpy as np } from '@jax-js/jax';

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
      acceptanceProb: np.array(0.5),
      isAccepted: np.array(true),
      isDivergent: np.array(false),
      energy: np.array(1.0),
      numIntegrationSteps: 10,
    };

    expect(info.momentum).toBeDefined();
    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.isDivergent).toBeDefined();
    expect(info.energy).toBeDefined();
    expect(info.numIntegrationSteps).toBeTypeOf('number');

    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('HMCConfig has required fields', () => {
    const config: HMCConfig = {
      stepSize: 0.1,
      numIntegrationSteps: 10,
      inverseMassMatrix: np.array([1.0]),
      divergenceThreshold: 1000,
    };

    expect(config.stepSize).toBeTypeOf('number');
    expect(config.numIntegrationSteps).toBeTypeOf('number');
    expect(config.inverseMassMatrix).toBeDefined();
    expect(config.divergenceThreshold).toBeTypeOf('number');

    config.inverseMassMatrix.dispose();
  });
});
