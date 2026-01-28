# Random Walk Metropolis (RWM) Sampler Design

**Date:** 2026-01-27
**Goal:** Add RWM sampler to core library for educational comparison with HMC in visualization

## Motivation

For educational purposes, having simpler MCMC algorithms alongside HMC demonstrates why HMC's gradient-informed proposals and momentum are valuable. RWM provides maximum pedagogical contrast: no gradients, no momentum, just random proposals.

## Directory Structure

```
src/rwm/
├── types.ts      # RWMState, RWMInfo, RWMConfig
├── kernel.ts     # createRWMKernel (the step logic)
├── builder.ts    # RWM() factory with fluent API
└── index.ts      # public exports

test/rwm/
├── types.test.ts      # Type contracts
├── builder.test.ts    # Builder API, immutability, validation
├── kernel.test.ts     # Step execution, proposal mechanics
├── refcount.test.ts   # Memory correctness (JAX-JS specific)
└── stationary.test.ts # Converges to correct distribution

examples/
└── memory-profile-rwm-jit-step.ts
```

## Types

```typescript
// src/rwm/types.ts

interface RWMState {
  position: Array;
  logdensity: Array;  // cached, no gradient needed
}

interface RWMInfo {
  acceptanceProb: Array;
  isAccepted: Array;
  proposedPosition: Array;  // useful for visualization
}

interface RWMConfig {
  logdensityFn: (position: Array) => Array;
  stepSize: number;
  jitStep?: boolean;  // default true
}

interface RWMSampler {
  init(position: Array): RWMState;
  step(key: Array, state: RWMState): [RWMState, RWMInfo];
}
```

**Key difference from HMC:** No `logdensityGrad` in state (RWM doesn't use gradients), no `isDivergent` or energy fields in info (no Hamiltonian to conserve).

## Kernel

```typescript
// src/rwm/kernel.ts

function createRWMKernel(config: RWMConfig) {
  const { logdensityFn, stepSize } = config;

  return function rwmStep(
    key: Array,
    state: RWMState
  ): [RWMState, RWMInfo] {
    // Split key for noise and acceptance
    const [noiseKey, acceptKey] = random.split(key);

    // 1. Propose: q' = q + ε * noise
    const noise = random.normal(noiseKey, state.position.shape);
    const proposedPosition = state.position.ref.add(noise.mul(stepSize));

    // 2. Evaluate log-density at proposal
    const proposedLogdensity = logdensityFn(proposedPosition.ref);

    // 3. Acceptance probability: min(1, exp(logπ(q') - logπ(q)))
    const logRatio = proposedLogdensity.ref.sub(state.logdensity.ref);
    const acceptanceProb = np.minimum(np.array(1.0), np.exp(logRatio));

    // 4. Accept/reject (branchless)
    const uniform = random.uniform(acceptKey, []);
    const isAccepted = uniform.lt(acceptanceProb.ref);

    const newPosition = np.where(isAccepted.ref, proposedPosition.ref, state.position.ref);
    const newLogdensity = np.where(isAccepted.ref, proposedLogdensity.ref, state.logdensity.ref);

    // 5. Dispose consumed arrays
    state.position.dispose();
    state.logdensity.dispose();
    proposedPosition.dispose();
    proposedLogdensity.dispose();

    const newState: RWMState = { position: newPosition, logdensity: newLogdensity };
    const info: RWMInfo = { acceptanceProb, isAccepted, proposedPosition: proposedPosition.ref };

    return [newState, info];
  };
}
```

**Memory pattern:** Same as HMC - consume inputs, dispose intermediates, return fresh state with refCount=1.

## Builder API

```typescript
// src/rwm/builder.ts

function RWM(logdensityFn: (position: Array) => Array): RWMBuilder {
  return new RWMBuilder({ logdensityFn });
}

class RWMBuilder {
  private config: Partial<RWMConfig>;

  constructor(config: Partial<RWMConfig>) {
    this.config = config;
  }

  stepSize(value: number): RWMBuilder {
    return new RWMBuilder({ ...this.config, stepSize: value });
  }

  jitStep(value: boolean = true): RWMBuilder {
    return new RWMBuilder({ ...this.config, jitStep: value });
  }

  build(): RWMSampler {
    // Validate required fields
    if (!this.config.logdensityFn) throw new Error('logdensityFn required');
    if (this.config.stepSize === undefined) throw new Error('stepSize required');

    const config: RWMConfig = {
      logdensityFn: this.config.logdensityFn,
      stepSize: this.config.stepSize,
      jitStep: this.config.jitStep ?? true,
    };

    const kernel = createRWMKernel(config);
    const step = config.jitStep ? jit(kernel) : kernel;

    return {
      init(position: Array): RWMState {
        return {
          position: position.ref,
          logdensity: config.logdensityFn(position),
        };
      },
      step,
    };
  }
}
```

**Usage:**
```typescript
const sampler = RWM(logdensityFn)
  .stepSize(0.1)
  .build();

const state = sampler.init(np.array([0.0, 0.0]));
const [newState, info] = sampler.step(random.key(42), state);
```

## Visualization Integration

**Changes to `examples/visualization/hmc-viz.ts`:**

1. Add algorithm dropdown to controls panel (alongside step size, integration steps)

2. Factory function to create sampler based on selection:
```typescript
function createSampler(
  algorithm: 'hmc' | 'rwm',
  logdensityFn: (q: Array) => Array,
  config: { stepSize: number; numIntegrationSteps?: number }
): Sampler {
  if (algorithm === 'rwm') {
    return RWM(logdensityFn).stepSize(config.stepSize).build();
  }
  return HMC(logdensityFn)
    .stepSize(config.stepSize)
    .numIntegrationSteps(config.numIntegrationSteps ?? 20)
    .build();
}
```

3. Hide "Integration Steps (L)" control when RWM is selected

4. Show "N/A" for energy/divergence stats when using RWM

## Testing Strategy

### Unit Tests

1. **Proposal mechanics:** Given fixed noise, verify `q' = q + ε * noise`

2. **Acceptance ratio:**
   - If proposal has higher density → acceptance prob = 1
   - If proposal has lower density → acceptance prob = exp(logπ(q') - logπ(q))

3. **Memory correctness:**
   - Old state consumed (refCount = 0, use-after-dispose throws)
   - New state fresh (refCount = 1)

4. **Stationary distribution:** Run 5000+ steps on 1D Gaussian, verify sample mean/variance converge to true values (with larger tolerance than HMC - RWM mixes slower)

5. **Builder validation:** Missing stepSize throws, immutability preserved

### Memory Profiling

**New script:** `examples/memory-profile-rwm-jit-step.ts`

**CI command:**
```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-rwm-jit-step.ts
```

**Expected behavior:** Memory should plateau below ~300MB, same as HMC.

## Implementation Order (TDD)

1. Types (RED → GREEN)
2. Kernel with memory tests (RED → GREEN)
3. Builder (RED → GREEN)
4. Stationary distribution test (RED → GREEN)
5. Memory profile script
6. Visualization integration

## Files to Modify

- `src/index.ts` - export RWM
- `examples/visualization/hmc-viz.ts` - add algorithm dropdown
- `CLAUDE.md` - add RWM memory check command
