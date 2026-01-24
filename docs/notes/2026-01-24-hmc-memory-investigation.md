# HMC Memory Investigation (2026-01-24)

## Summary
We profiled HMC memory growth under three execution modes using the examples in
`examples/` with `NODE_OPTIONS="--expose-gc"`:

- **Eager HMC** grows rapidly into GBs over a few hundred iterations.
- **HMC with `valueAndGrad({ jit: true })`** still grows rapidly.
- **HMC with `.jitStep()`** stays flat over long runs (2,000 iterations).

This strongly suggests the memory growth is dominated by the *eager* execution
path (many intermediate arrays and kernels per step), not by HMC state retention
or cache growth from JIT compilation.

## Benchmarks
All runs used the same target distribution (1D standard normal) and default
example settings unless noted. Each run used the instrumented JAX-JS runtime
and logged memory every N iterations.

### 1) HMC + `jitStep()` (stable)
Command:
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
```
Output:
```
start heap=8.4MB rss=108.1MB
iter 500 heap=23.6MB rss=220.2MB
iter 1000 heap=35.9MB rss=239.1MB
iter 1500 heap=67.3MB rss=268.0MB
iter 2000 heap=62.6MB rss=260.1MB
end heap=62.5MB rss=260.1MB
```

Longer attempt (5,000 iterations) hit a JAX-JS internal stack overflow after
2,000 iterations (see “Known limitations” below), but memory remained stable
up to that point.

### 2) HMC eager (grows fast)
Command:
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=400 LOG_EVERY=100 npx tsx examples/memory-profile-hmc.ts
```
Partial output (run timed out after ~200 iters):
```
start heap=8.4MB rss=109.0MB
iter 100 heap=283.0MB rss=551.4MB
iter 200 heap=1060.2MB rss=1368.9MB
```

### 3) HMC + `valueAndGrad({ jit: true })` (still grows)
Command:
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=300 LOG_EVERY=100 npx tsx examples/memory-profile-hmc-jit-value-and-grad.ts
```
Output:
```
start heap=8.7MB rss=109.5MB
iter 100 heap=167.0MB rss=429.0MB
iter 200 heap=604.6MB rss=888.3MB
iter 300 heap=1188.6MB rss=1484.4MB
end heap=1188.5MB rss=1484.6MB
```

## Findings
1. **`.jitStep()` stabilizes memory** over thousands of iterations.
2. **`valueAndGrad({ jit: true })` is not sufficient**; eager mode still grows.
3. **JIT cache growth is tiny** (`jitCompileCache` stayed at 1–3 entries) during
   these runs, and no wasm module cache growth was observed.

## Root-Cause Hypotheses (evidence-based)
These are informed by the profiling above but should be treated as hypotheses
until deeper instrumentation is added:

1. **Eager execution allocates many intermediate arrays** each step. Even if
   refcounts are correct, the wasm allocator’s high-water mark can rise, and
   `WebAssembly.Memory` does not shrink.
2. **Kernel fusion in `jitStep()`** drastically reduces intermediate allocations,
   keeping memory bounded.
3. **Cache growth is unlikely to be the primary driver** in these runs; the
   observed `jitCompileCache` growth is tiny, and we did not see `wasm.moduleCache`
   growth logs during the tests.

## Debug Tools (kept for future profiling)
### 1) Example scripts
- `examples/memory-profile-hmc.ts` (eager)
- `examples/memory-profile-hmc-jit-value-and-grad.ts`
- `examples/memory-profile-hmc-jit-step.ts`
- `examples/memory-profile-gradient-descent.ts`

### 2) Loader to use instrumented JAX-JS
Use `tools/jaxjs-loader.mjs` to redirect `@jax-js/jax` to `/tmp/jax-js`:
```bash
NODE_OPTIONS="--loader ./tools/jaxjs-loader.mjs"
```

### 3) Cache logging
Set `JAXJS_CACHE_LOG=1` to enable cache logs. This requires an instrumented
`/tmp/jax-js` checkout (see below).

### 4) Instrumented JAX-JS notes
Our local `/tmp/jax-js` was modified to log cache sizes when
`JAXJS_CACHE_LOG=1` is set. The logging helpers live in:
- `src/frontend/jit.ts` (jitCompileCache)
- `src/frontend/jvp.ts` (jvpJaxprCache)
- `src/frontend/linearize.ts` (transposeJaxprCache)
- `src/frontend/vmap.ts` (vmapJaxprCache)
- `src/backend/wasm.ts` (wasm.moduleCache)
- `src/backend/webgpu.ts` (shader/pipeline caches)
- `src/backend/webgl.ts` (program cache)

If `/tmp/jax-js` is re-cloned, reapply those logging snippets or use a branch
that already contains them.

## Known limitations
- **5000-iteration `jitStep()` run hit an internal stack overflow** in
  `/tmp/jax-js/src/frontend/array.ts`:
  `RangeError: Maximum call stack size exceeded` around ~2,000 iterations.
  This is likely an internal pending-kernel dispatch recursion and not directly
  tied to HMC.

## Next steps (optional)
- Add allocator-level stats (live buffer count/bytes) to confirm the high-water
  hypothesis.
- Investigate the `pending.splice` recursion in JAX-JS and add a non-recursive
  queue drain.
