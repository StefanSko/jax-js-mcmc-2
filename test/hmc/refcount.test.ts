import { describe, it, expect } from 'vitest';
import { numpy as np, grad, random } from '@jax-js/jax';
import { createHMCKernel } from '../../src/hmc/kernel';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { HMCConfig, HMCInfo, HMCState } from '../../src/hmc/types';

const logdensityFn = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

const createConfig = (): HMCConfig => ({
  stepSize: 0.1,
  numIntegrationSteps: 5,
  inverseMassMatrix: np.array([1.0]),
  divergenceThreshold: 1000,
});

const createInitialState = (): HMCState => {
  const position = np.array([0.0]);
  const positionRef = position.ref;
  position.dispose();
  return {
    position: positionRef,
    logdensity: logdensityFn(positionRef.ref),
    logdensityGrad: grad(logdensityFn)(positionRef.ref),
  };
};

const disposeInfo = (info: HMCInfo): void => {
  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();
};

describe('HMC kernel refcount', () => {
  it('old state arrays have refCount 0 after step', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const oldPosition = state.position;
    const oldLogdensity = state.logdensity;
    const oldLogdensityGrad = state.logdensityGrad;

    const key = random.key(42);
    const [newState, info] = hmcStep(key, state);

    expect(oldPosition.refCount).toBe(0);
    expect(oldLogdensity.refCount).toBe(0);
    expect(oldLogdensityGrad.refCount).toBe(0);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    disposeInfo(info);
    config.inverseMassMatrix.dispose();
  });

  it('new state arrays have refCount 1 after step', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const key = random.key(42);
    const [newState, info] = hmcStep(key, state);

    expect(newState.position.refCount).toBe(1);
    expect(newState.logdensity.refCount).toBe(1);
    expect(newState.logdensityGrad.refCount).toBe(1);

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    disposeInfo(info);
    config.inverseMassMatrix.dispose();
  });

  it('no reference accumulation over many HMC steps', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    let state = createInitialState();

    for (let i = 0; i < 50; i++) {
      const oldPosition = state.position;
      const key = random.key(i);
      const [newState, info] = hmcStep(key, state);

      expect(oldPosition.refCount).toBe(0);
      expect(newState.position.refCount).toBe(1);

      disposeInfo(info);
      state = newState;
    }

    expect(state.position.refCount).toBe(1);

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
    config.inverseMassMatrix.dispose();
  });
});
