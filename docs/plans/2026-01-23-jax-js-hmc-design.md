# JAX-JS HMC Sampler Design

## Overview

A Hamiltonian Monte Carlo (HMC) sampler built on JAX-JS, with careful attention to JAX-JS's unique memory management model (reference counting with `.ref` pattern). Inspired by BlackJAX's architecture.

**Goals:**
- Full HMC with diagnostics (divergence detection, acceptance rate, HMCInfo)
- Strong test suite: physics-inspired tests + JAX-JS memory stress tests
- Strict TDD workflow with interleaved tests
- TypeScript-first with strict typing and linting

## Project Structure

```
jax-js-mcmc/
├── src/
│   ├── index.ts                 # Public exports
│   ├── integrators/
│   │   ├── types.ts             # IntegratorState, Integrator type
│   │   ├── velocity-verlet.ts   # Phase 1
│   │   ├── mclachlan.ts         # Phase 4
│   │   └── yoshida.ts           # Phase 4
│   ├── hmc/
│   │   ├── types.ts             # HMCState, HMCInfo, HMCConfig
│   │   ├── kernel.ts            # Functional core (hmcStep)
│   │   ├── proposal.ts          # Propose + accept/reject logic
│   │   └── builder.ts           # Immutable builder API
│   └── metrics/
│       ├── types.ts             # Metric interface
│       └── gaussian-euclidean.ts # Kinetic energy, momentum sampling
├── test/
│   ├── setup.ts                 # Vitest config, custom matchers
│   ├── integrators/
│   │   ├── types.test.ts        # Type contracts
│   │   ├── velocity-verlet.test.ts
│   │   └── refcount.test.ts     # Memory correctness
│   ├── physics/
│   │   ├── harmonic-oscillator.test.ts
│   │   └── energy-conservation.test.ts
│   ├── hmc/
│   │   ├── types.test.ts
│   │   ├── kernel.test.ts
│   │   ├── refcount.test.ts
│   │   ├── stationary.test.ts
│   │   └── info.test.ts
│   └── memory/
│       ├── gradient-flow.test.ts
│       └── divergence.test.ts
├── package.json
├── tsconfig.json
├── eslint.config.ts
└── vitest.config.ts
```

## Core Types

### Integrator Types

```typescript
// src/integrators/types.ts
import { Array } from "@jax-js/jax";

export interface IntegratorState {
  position: Array;
  momentum: Array;
  logdensity: Array;        // Scalar - cached to avoid recomputation
  logdensityGrad: Array;    // Gradient at current position
}

export type Integrator = (
  state: IntegratorState,
  stepSize: number
) => IntegratorState;
```

### HMC Types

```typescript
// src/hmc/types.ts
import { Array } from "@jax-js/jax";
import { Integrator } from "../integrators/types";

export interface HMCState {
  position: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export interface HMCInfo {
  momentum: Array;          // The sampled momentum (for diagnostics)
  acceptanceRate: number;   // Probability of accepting proposal
  isAccepted: boolean;      // Whether proposal was accepted
  isDivergent: boolean;     // Energy difference exceeded threshold
  energy: number;           // Total Hamiltonian energy
  numIntegrationSteps: number;
}

export interface HMCConfig {
  stepSize: number;
  numIntegrationSteps: number;
  inverseMassMatrix: Array;
  divergenceThreshold: number;  // Default: 1000
  integrator: Integrator;       // Default: velocityVerlet
}
```

## Key Implementation Details

### Velocity Verlet Integrator

```typescript
// src/integrators/velocity-verlet.ts
export function createVelocityVerlet(
  logdensityFn: (position: Array) => Array,
  kineticEnergyGradFn: (momentum: Array) => Array
): Integrator {
  return function velocityVerletStep(
    state: IntegratorState,
    stepSize: number
  ): IntegratorState {
    const halfStep = stepSize * 0.5;

    // Half step momentum: p += (ε/2) * ∇log π(q)
    const momentumHalf = state.momentum.add(
      state.logdensityGrad.ref.mul(halfStep)
    );

    // Full step position: q += ε * ∇K(p)
    const kineticGrad = kineticEnergyGradFn(momentumHalf.ref);
    const newPosition = state.position.add(kineticGrad.mul(stepSize));

    // Compute new gradient
    const [newLogdensity, newLogdensityGrad] = logdensityAndGrad(newPosition.ref);

    // Half step momentum: p += (ε/2) * ∇log π(q_new)
    const newMomentum = momentumHalf.add(newLogdensityGrad.ref.mul(halfStep));

    // Dispose old state arrays (consumed)
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();

    return {
      position: newPosition,
      momentum: newMomentum,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };
  };
}
```

### JIT-Unrolled Integration Loop

JAX-JS doesn't have `fori_loop`/`scan`, so we unroll at JIT time:

```typescript
export function createHMCKernel(config: HMCConfig, logdensityFn, metric) {
  // Unroll integration loop at JIT compile time
  const integrateTrajectory = jit((state: IntegratorState) => {
    let s = state;
    for (let i = 0; i < config.numIntegrationSteps; i++) {
      s = config.integrator(s, config.stepSize);
    }
    return s;
  });

  return function hmcStep(key, state) {
    // ... momentum sampling
    const finalState = integrateTrajectory(integState);
    // ... accept/reject
  };
}
```

### Branchless Accept/Reject with np.where

```typescript
// src/hmc/proposal.ts
export function selectState(
  originalState: HMCState,
  proposalIntegState: IntegratorState,
  isAccepted: Array,
  // ... other params
): [HMCState, HMCInfo] {

  // Branchless selection with np.where
  const newPosition = np.where(
    isAccepted.ref,
    proposalIntegState.position.ref,
    originalState.position.ref
  );
  const newLogdensity = np.where(
    isAccepted.ref,
    proposalIntegState.logdensity.ref,
    originalState.logdensity.ref
  );
  const newLogdensityGrad = np.where(
    isAccepted.ref,
    proposalIntegState.logdensityGrad.ref,
    originalState.logdensityGrad.ref
  );

  // Dispose ALL inputs uniformly - no conditional disposal needed
  originalState.position.dispose();
  originalState.logdensity.dispose();
  originalState.logdensityGrad.dispose();
  proposalIntegState.position.dispose();
  proposalIntegState.logdensity.dispose();
  proposalIntegState.logdensityGrad.dispose();

  return [
    { position: newPosition, logdensity: newLogdensity, logdensityGrad: newLogdensityGrad },
    info
  ];
}
```

### Immutable Builder API

```typescript
// src/hmc/builder.ts
export class HMCBuilder {
  private constructor(
    private readonly logdensityFn: (pos: Array) => Array,
    private readonly config: Partial<HMCConfig>
  ) {}

  static create(logdensityFn: (pos: Array) => Array): HMCBuilder {
    return new HMCBuilder(logdensityFn, {});
  }

  stepSize(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, { ...this.config, stepSize: value });
  }

  numIntegrationSteps(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, { ...this.config, numIntegrationSteps: value });
  }

  inverseMassMatrix(value: Array): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, { ...this.config, inverseMassMatrix: value });
  }

  divergenceThreshold(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, { ...this.config, divergenceThreshold: value });
  }

  build(): HMCSampler {
    const config = this.validateAndFillDefaults();
    const metric = createGaussianEuclidean(config.inverseMassMatrix);
    const integrator = createVelocityVerlet(this.logdensityFn, metric.kineticEnergyGrad);

    return {
      init: (position: Array) => initHMCState(position, this.logdensityFn),
      step: createHMCKernel({ ...config, integrator }, this.logdensityFn, metric),
    };
  }
}

// Convenience export
export const HMC = HMCBuilder.create;
```

**Usage:**
```typescript
const sampler = HMC(logdensityFn)
  .stepSize(0.1)
  .numIntegrationSteps(10)
  .inverseMassMatrix(np.ones([dim]))
  .build();

const state = sampler.init(initialPosition);
const [newState, info] = sampler.step(key, state);
```

## Testing Strategy

### Physics Tests (from BlackJAX)

**Harmonic Oscillator:**
- Potential: U(q) = 0.5 * k * q²
- Exact solution: q(t) = q₀cos(ωt) + p₀sin(ωt)/ω
- Test: trajectory matches analytical solution

**Energy Conservation:**
- Run many integration steps
- Verify |E_final - E_initial| < tolerance (1e-4)

### Memory Stress Tests (JAX-JS specific)

**Priority failure modes:**
1. Double consumption - array used twice without `.ref`
2. Incorrect gradient flow - `.ref` breaks autodiff chain
3. Leaked arrays - memory grows unbounded
4. Premature disposal - array freed while still needed

**RefCount validation tests:**
```typescript
test("HMC state arrays have refCount 1 after step", () => {
  const [newState, info] = sampler.step(key, state);

  // Old state consumed (refCount 0)
  expect(state.position.refCount).toBe(0);

  // New state fresh (refCount 1)
  expect(newState.position.refCount).toBe(1);
});

test("old state arrays throw after being consumed", () => {
  const [newState, _] = sampler.step(key, state);

  // Use-after-dispose throws ReferenceError
  expect(() => state.position.js()).toThrowError(ReferenceError);
});

test("no accumulating references over many iterations", () => {
  for (let i = 0; i < 1000; i++) {
    const [newState, info] = sampler.step(key, state);
    expect(state.position.refCount).toBe(0);
    expect(newState.position.refCount).toBe(1);
    state = newState;
  }
});
```

## TDD Implementation Phases

| Phase | Write Tests First (RED) | Then Implement (GREEN) |
|-------|------------------------|------------------------|
| **1a** | `integrator.types.test.ts` - type contracts | `types.ts` |
| **1b** | `harmonic-oscillator.test.ts` - trajectory | `velocity-verlet.ts` |
| **1c** | `energy-conservation.test.ts` - ΔE < tol | Fix drift bugs |
| **1d** | `integrator-refcount.test.ts` - memory | Fix .ref/.dispose |
| **2a** | `hmc.types.test.ts` - type contracts | `hmc/types.ts` |
| **2b** | `hmc-kernel.test.ts` - runs without error | `kernel.ts` skeleton |
| **2c** | `hmc-acceptance.test.ts` - accept/reject | Metropolis-Hastings |
| **2d** | `hmc-refcount.test.ts` - no leaks | `selectState` memory |
| **2e** | `hmc-stationary.test.ts` - correct dist | Full kernel |
| **3a** | `divergence.test.ts` - detects jumps | Divergence tracking |
| **3b** | `gradient-flow.test.ts` - autodiff works | Verify chain |
| **3c** | `hmc-info.test.ts` - diagnostics | HMCInfo complete |
| **4** | Comparative accuracy tests | McLachlan/Yoshida |
| **5** | Ill-conditioned Gaussian | Dense mass matrix |

## Tooling Configuration

### TypeScript (strict)

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "exactOptionalPropertyTypes": true,
    "noUncheckedIndexedAccess": true,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  }
}
```

### ESLint

```typescript
// eslint.config.ts
import tseslint from 'typescript-eslint';

export default tseslint.config(
  tseslint.configs.strictTypeChecked,
  {
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
    }
  }
);
```

### Vitest (with typecheck)

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    typecheck: {
      enabled: true,
      tsconfig: './tsconfig.json',
    },
    globals: true,
    setupFiles: ['./test/setup.ts'],
  },
});
```

### Package Scripts

```json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint src test --ext .ts",
    "lint:fix": "eslint src test --ext .ts --fix",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:types": "tsc --noEmit && vitest run",
    "ci": "npm run typecheck && npm run lint && npm run test"
  }
}
```

## JAX-JS Memory Patterns Summary

| Pattern | When to Use |
|---------|-------------|
| `x.ref` | Need to use `x` again after this operation |
| `x.dispose()` | Done with `x`, decrement refCount |
| `np.where(cond, a, b)` | Branchless conditional (both branches computed) |
| Check `x.refCount` | Debug/test memory correctness |
| `ReferenceError` | Thrown on use-after-dispose |

**Rule:** Every array entering a function is either:
1. Consumed by an operation (refCount decremented automatically)
2. Used with `.ref` if needed multiple times
3. Explicitly `.dispose()`d if unused

## References

- [JAX-JS Repository](https://github.com/ekzhang/jax-js)
- [BlackJAX Repository](https://github.com/blackjax-devs/blackjax)
- [JAX-JS Memory Management Analysis](../jax-js-memory-management-analysis.md)
