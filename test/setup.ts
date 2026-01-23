import { numpy as np } from '@jax-js/jax';
import { expect } from 'vitest';

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ) {
    const { isNot } = this;
    const actualArray = np.array(actual);
    const expectedArray = np.array(expected);
    return {
      pass: np.allclose(actualArray.ref, expectedArray.ref, options),
      message: () => `expected array to be${isNot ? ' not' : ''} allclose`,
      actual: actualArray.js(),
      expected: expectedArray.js(),
    };
  },
});

declare module 'vitest' {
  interface Assertion<T> {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
  interface AsymmetricMatchersContaining {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
}
