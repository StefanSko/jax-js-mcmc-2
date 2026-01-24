// Node ESM loader to redirect @jax-js/jax to a local /tmp checkout.
// Useful for profiling against an instrumented JAX-JS runtime.
export async function resolve(specifier, context, nextResolve) {
  if (specifier === '@jax-js/jax') {
    return {
      url: new URL('file:///tmp/jax-js/src/index.ts').href,
      shortCircuit: true,
    };
  }
  return nextResolve(specifier, context);
}
