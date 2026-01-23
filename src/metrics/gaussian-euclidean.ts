import { grad, numpy as np, random, type Array } from '@jax-js/jax';
import type { Metric } from './types';

export function createGaussianEuclidean(inverseMassMatrix: Array): Metric {
  const massMatrixSqrt = np.sqrt(np.reciprocal(inverseMassMatrix.ref));

  const kineticEnergy = (momentum: Array): Array => {
    const scaled = momentum.ref.mul(momentum).mul(inverseMassMatrix.ref);
    const result = scaled.sum().mul(0.5);
    return result;
  };

  const kineticEnergyGrad = grad(kineticEnergy);

  const sampleMomentum = (key: Array, position: Array): Array => {
    const z = random.normal(key, position.shape);
    const momentum = z.mul(massMatrixSqrt.ref);
    position.dispose();
    return momentum;
  };

  return {
    sampleMomentum,
    kineticEnergy,
    kineticEnergyGrad,
  };
}
