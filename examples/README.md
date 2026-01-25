# Examples

These scripts demonstrate basic usage of the HMC sampler. Sample counts are
intentionally small to keep memory use manageable with current JAX-JS behavior.

Memory profiling helpers:
- `examples/memory-profile-hmc.ts` logs heap/RSS while running HMC (forces eager
  mode via `.jitStep(false)`).
- `examples/memory-profile-hmc-jit-step.ts` jit-compiles the entire HMC step
  (default behavior).
- `examples/memory-profile-gradient-descent.ts` does the same for a simple
  gradient descent loop.

Tip: run with `NODE_OPTIONS="--expose-gc"` to force GC before each log.

Debugging caches (requires an instrumented `/tmp/jax-js` checkout):
- Use `tools/jaxjs-loader.mjs` to redirect `@jax-js/jax` to `/tmp/jax-js`.
- Enable cache logging with `JAXJS_CACHE_LOG=1`.
Example:
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \\
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
```

Run with a TypeScript runner (e.g. `tsx` or `ts-node`) or copy snippets into your app.
