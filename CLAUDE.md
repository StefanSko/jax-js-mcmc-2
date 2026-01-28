# JAX-JS HMC Sampler

Building an HMC sampler on JAX-JS with strict TDD and memory-safe patterns.

## Development Workflow

**Strict TDD (RED/GREEN):**
```
1. Write failing test (RED)  → test fails or doesn't compile
2. Write minimal code (GREEN) → test passes
3. Refactor                   → test still passes
4. Next test
```

**Every PR must pass:**
```bash
npm run ci  # typecheck + lint + test
```

**Mandatory integration check (memory):**
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
```
**Desired output:** memory should plateau (heap and rss do not trend upward) and
stay comfortably below ~300MB by the end of 2,000 iterations. If it grows
monotonically or exceeds ~500MB, treat as a regression.

**RWM memory check:**
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-rwm-jit-step.ts
```

**Why prefer JIT mode:** eager mode executes each primitive immediately and
allocates many intermediate arrays per step, which drives the allocator's
high-water mark upward. `jitStep()` fuses the whole step into a compiled kernel,
dramatically reducing intermediate allocations and stabilizing memory.


## JAX-JS Memory Model

JAX-JS uses reference counting. Every array has a refCount.

| Pattern | Meaning |
|---------|---------|
| `x.ref` | Increment refCount, use x again later |
| `x.dispose()` | Decrement refCount, done with x |
| `x.refCount` | Check current count (for tests) |

**Rule:** Functions consume their inputs. If you need an input again, use `.ref`:
```typescript
// WRONG - double consumption
const y = x.add(1);
const z = x.mul(2);  // Error: x already consumed

// CORRECT
const y = x.ref.add(1);
const z = x.mul(2);  // x consumed here
```

**Testing memory correctness:**
```typescript
expect(oldState.position.refCount).toBe(0);  // consumed
expect(newState.position.refCount).toBe(1);  // fresh
expect(() => oldState.position.js()).toThrowError(ReferenceError);
```

## Reference Repositories

**JAX-JS** (the runtime): `/tmp/jax-js`
- Pull latest: `cd /tmp/jax-js && git pull`
- Key files: `src/frontend/array.ts`, `src/library/random.ts`, `src/frontend/jvp.ts`
- Tests: `test/*.test.ts` (vitest, custom `toBeAllclose` matcher)

**BlackJAX** (reference HMC implementation): `/tmp/blackjax`
- Pull latest: `cd /tmp/blackjax && git pull`
- HMC: `blackjax/mcmc/hmc.py`, `blackjax/mcmc/integrators.py`
- Tests: `tests/mcmc/test_integrators.py` (physics-inspired tests)

## Key Design Decisions

1. **Immutable builder API** - `HMC(logdensity).stepSize(0.1).build()`
2. **JIT-unrolled loops** - No `fori_loop` in JAX-JS, use for-loop inside `jit()`
3. **Branchless accept/reject** - Use `np.where()`, dispose all inputs uniformly
4. **Physics tests** - Harmonic oscillator, energy conservation
5. **Memory tests** - refCount validation, gradient flow, use-after-dispose

## Current Phase

See `docs/plans/2026-01-23-jax-js-hmc-design.md` for full design and phase breakdown.
