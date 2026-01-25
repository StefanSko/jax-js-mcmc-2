import { grad, numpy as np } from '@jax-js/jax';
import {
  createGaussian2D,
  createBanana,
  createFunnel,
} from '../../examples/visualization/distributions';

describe('visualization distributions', () => {
  test('logdensity evaluates with integer indices', () => {
    const distributions = [createGaussian2D(), createBanana(), createFunnel()];

    for (const dist of distributions) {
      const position = np.array([0, 0]);
      const value = dist.logdensity(position);
      expect(value.shape).toEqual([]);
      value.dispose();
    }
  });

  test('logdensity supports reverse-mode grad', () => {
    const distributions = [createGaussian2D(), createBanana(), createFunnel()];

    for (const dist of distributions) {
      const position = np.array([0, 0]);
      const valueGrad = grad(dist.logdensity)(position);
      expect(valueGrad.shape).toEqual([2]);
      valueGrad.dispose();
    }
  });
});
