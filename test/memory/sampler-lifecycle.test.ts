import { describe, expect, it } from 'vitest';
import { numpy as np } from '@jax-js/jax';
import { HMC } from '../../src/hmc/builder';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';

const logdensityFn = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

describe('sampler lifecycle memory ownership', () => {
  it('Gaussian metric dispose releases owned mass-matrix arrays', () => {
    const inverseMassMatrix = np.array([1.0, 2.0, 3.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    expect(inverseMassMatrix.refCount).toBe(1);

    metric.dispose();

    expect(inverseMassMatrix.refCount).toBe(0);
    expect(() => {
      inverseMassMatrix.js();
    }).toThrowError(ReferenceError);
  });

  it('HMC sampler dispose releases the inverse mass matrix passed to the builder', () => {
    const inverseMassMatrix = np.array([1.0]);
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(3)
      .inverseMassMatrix(inverseMassMatrix)
      .build();

    expect(inverseMassMatrix.refCount).toBe(1);

    sampler.dispose();

    expect(inverseMassMatrix.refCount).toBe(0);
    expect(() => {
      inverseMassMatrix.js();
    }).toThrowError(ReferenceError);
  });

  it('HMC sampler dispose is idempotent', () => {
    const inverseMassMatrix = np.array([1.0]);
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(3)
      .inverseMassMatrix(inverseMassMatrix)
      .build();

    sampler.dispose();
    sampler.dispose();

    expect(inverseMassMatrix.refCount).toBe(0);
  });
});
