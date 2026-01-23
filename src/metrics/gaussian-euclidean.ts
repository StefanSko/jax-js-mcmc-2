import { grad, numpy as np, random, type Array } from '@jax-js/jax';
import type { Metric } from './types';

export function createGaussianEuclidean(inverseMassMatrix: Array): Metric {
  // For diagonal mass matrix: M^{-1} = diag(inverseMassMatrix)
  // Mass matrix sqrt: M^{1/2} = diag(1/sqrt(inverseMassMatrix))
  // M^{1/2} = sqrt(1 / M^{-1}) = sqrt(M)
  const massMatrixSqrt = np.sqrt(np.reciprocal(inverseMassMatrix.ref));

  const kineticEnergy = (momentum: Array): Array => {
    // K = 0.5 * p^T * M^{-1} * p = 0.5 * sum(p^2 * M^{-1})
    const scaled = momentum.ref.mul(momentum).mul(inverseMassMatrix.ref);
    return np.sum(scaled).mul(0.5);
  };

  const kineticEnergyGrad = grad(kineticEnergy);

  const sampleMomentum = (key: Array, position: Array): Array => {
    // p ~ N(0, M) => p = M^{1/2} * z where z ~ N(0, I)
    const shape = position.shape;
    position.dispose(); // Consume position - we only needed its shape
    const z = random.normal(key, shape);
    return z.mul(massMatrixSqrt.ref);
  };

  return {
    sampleMomentum,
    kineticEnergy,
    kineticEnergyGrad,
  };
}
