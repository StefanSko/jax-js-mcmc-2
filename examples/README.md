# Examples

These scripts demonstrate basic usage of the HMC sampler. Sample counts are
intentionally small to keep memory use manageable with current JAX-JS behavior.

Memory profiling helpers:
- `examples/memory-profile-hmc.ts` logs heap/RSS while running HMC.
- `examples/memory-profile-hmc-jit-value-and-grad.ts` uses a JIT-compiled
  value-and-grad path for logdensity in the integrator.
- `examples/memory-profile-gradient-descent.ts` does the same for a simple
  gradient descent loop.

Tip: run with `NODE_OPTIONS="--expose-gc"` to force GC before each log.

Run with a TypeScript runner (e.g. `tsx` or `ts-node`) or copy snippets into your app.
