import { Array as JaxArray, numpy as np } from '@jax-js/jax';
import type { ExpectationResult } from '@vitest/expect';
import { expect } from 'vitest';

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ): ExpectationResult {
    const { isNot } = this;
    const actualInput = actual instanceof JaxArray ? actual.ref : actual;
    const expectedInput = expected instanceof JaxArray ? expected.ref : expected;
    const actualArray = np.array(actualInput);
    const expectedArray = np.array(expectedInput);
    const pass = np.allclose(actualArray.ref, expectedArray.ref, options);
    actualArray.dispose();
    expectedArray.dispose();
    return {
      pass,
      message: () => `expected array to be${isNot ? ' not' : ''} allclose`,
      actual: actual,
      expected: expected,
    };
  },
});

declare module 'vitest' {
  interface Assertion<T> {
    toBeAllclose(
      expected: np.ArrayLike,
      options?: { rtol?: number; atol?: number }
    ): void;
    readonly _type?: T;
  }
  interface AsymmetricMatchersContaining {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
}
