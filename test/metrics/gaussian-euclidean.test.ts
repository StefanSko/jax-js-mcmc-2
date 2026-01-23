import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@jax-js/jax';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';

describe('Gaussian Euclidean Metric', () => {
  it('sampleMomentum returns array with correct shape', () => {
    const inverseMassMatrix = np.array([1.0, 1.0, 1.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const key = random.key(42);
    const position = np.array([0.0, 0.0, 0.0]);
    const momentum = metric.sampleMomentum(key, position);

    expect(momentum.shape).toEqual([3]);

    momentum.dispose();
    position.dispose();
    inverseMassMatrix.dispose();
  });

  it('kineticEnergy computes 0.5 * p^T * M^{-1} * p', () => {
    const inverseMassMatrix = np.array([1.0, 2.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const momentum = np.array([2.0, 3.0]);
    const energy = metric.kineticEnergy(momentum);

    expect(energy).toBeAllclose(11.0);

    energy.dispose();
    inverseMassMatrix.dispose();
  });

  it('kineticEnergyGrad computes M^{-1} * p', () => {
    const inverseMassMatrix = np.array([1.0, 2.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const momentum = np.array([2.0, 3.0]);
    const grad = metric.kineticEnergyGrad(momentum);

    expect(grad).toBeAllclose([2.0, 6.0]);

    grad.dispose();
    inverseMassMatrix.dispose();
  });
});
