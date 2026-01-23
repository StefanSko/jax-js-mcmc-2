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

    // Both .js() and allclose consume arrays via dataSync().
    // We need extra refs to survive both operations AND preserve the original.
    // Add 2 refs: one for .js(), one for allclose (original stays at refCount 1)
    void actualArray.ref;  // +1 for .js()
    void actualArray.ref;  // +1 for allclose
    void expectedArray.ref;  // +1 for .js()
    void expectedArray.ref;  // +1 for allclose

    // Get JS values (consumes 1 ref each)
    const actualJs = actualArray.js() as unknown;
    const expectedJs = expectedArray.js() as unknown;

    // allclose consumes 1 ref each
    const pass = np.allclose(actualArray, expectedArray, options);

    return {
      pass,
      message: (): string => `expected array to be${isNot ? ' not' : ''} allclose`,
      actual: actualJs,
      expected: expectedJs,
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
