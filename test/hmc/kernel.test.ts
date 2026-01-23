import { describe, it, expect } from 'vitest';
import { numpy as np, grad, random } from '@jax-js/jax';
import { createHMCKernel } from '../../src/hmc/kernel';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { HMCState, HMCConfig } from '../../src/hmc/types';

describe('HMC Kernel', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    // Standard normal: log p(q) = -0.5 * q^2
    return q.ref.mul(q).mul(-0.5).sum();
  };

  const createConfig = (): HMCConfig => ({
    stepSize: 0.1,
    numIntegrationSteps: 10,
    inverseMassMatrix: np.array([1.0]),
    divergenceThreshold: 1000,
  });

  const createInitialState = (): HMCState => {
    const position = np.array([0.0]);
    return {
      position: position.ref,
      logdensity: logdensityFn(position.ref),
      logdensityGrad: grad(logdensityFn)(position),
    };
  };

  it('returns new state and info', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const key = random.key(42);

    const [newState, info] = hmcStep(key, state);

    // New state should have all required fields
    expect(newState.position).toBeDefined();
    expect(newState.logdensity).toBeDefined();
    expect(newState.logdensityGrad).toBeDefined();

    // Info should have all required fields
    expect(info.momentum).toBeDefined();
    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.isDivergent).toBeDefined();
    expect(info.energy).toBeDefined();
    expect(info.numIntegrationSteps).toBe(10);

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
    config.inverseMassMatrix.dispose();
  });
});
