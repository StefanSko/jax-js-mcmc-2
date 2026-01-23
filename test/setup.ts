import { numpy as np } from '@jax-js/jax';
import { expect } from 'vitest';

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ): { pass: boolean; message: () => string; actual: unknown; expected: unknown } {
    const { isNot } = this;
    const actualArray = np.array(actual);
    const expectedArray = np.array(expected);
    return {
      pass: np.allclose(actualArray.ref, expectedArray.ref, options),
      message: (): string => `expected array to be${isNot ? ' not' : ''} allclose`,
      actual: actualArray.js() as unknown,
      expected: expectedArray.js() as unknown,
    };
  },
});

declare module 'vitest' {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface Assertion<T> {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
  interface AsymmetricMatchersContaining {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
}
