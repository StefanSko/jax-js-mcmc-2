# JAX-JS HMC Sampler

A Hamiltonian Monte Carlo (HMC) sampler built on [JAX-JS](https://github.com/anthropics/jax-js) with strict memory management and physics-validated implementations.

## Installation

```bash
npm install
```

## Quick Start

```typescript
import { numpy as np, random } from '@jax-js/jax';
import { HMC } from 'jax-js-mcmc';

// Define target distribution: standard normal N(0, 1)
const logdensity = (q: np.Array): np.Array => {
  return q.ref.mul(q).mul(-0.5).sum();
};

// Build sampler
const sampler = HMC(logdensity)
  .stepSize(0.1)
  .numIntegrationSteps(10)
  .inverseMassMatrix(np.array([1.0]))
  .build();

// Initialize state
let state = sampler.init(np.array([0.0]));

// Run sampling
for (let i = 0; i < 1000; i++) {
  const key = random.key(i);
  const [newState, info] = sampler.step(key, state);

  // Process sample...
  console.log('Position:', newState.position.ref.js());

  // Clean up info arrays
  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();

  state = newState;
}

// Clean up final state
state.position.dispose();
state.logdensity.dispose();
state.logdensityGrad.dispose();
```

## API Reference

### HMC Builder

Create an HMC sampler using the fluent builder API:

```typescript
const sampler = HMC(logdensityFn)
  .stepSize(0.1)              // Integration step size (required)
  .numIntegrationSteps(10)    // Leapfrog steps per iteration (required)
  .inverseMassMatrix(M)       // Diagonal inverse mass matrix (required)
  .divergenceThreshold(1000)  // Energy threshold for divergence (default: 1000)
  .build();
```

### Sampler Methods

#### `sampler.init(position: Array): HMCState`

Initialize the sampler state from a starting position.

```typescript
const state = sampler.init(np.array([0.0, 0.0]));
```

#### `sampler.step(key: Array, state: HMCState): [HMCState, HMCInfo]`

Perform one HMC step, returning the new state and diagnostic info.

```typescript
const key = random.key(42);
const [newState, info] = sampler.step(key, state);
```

### Types

#### HMCState
```typescript
interface HMCState {
  position: Array;       // Current position
  logdensity: Array;     // Log density at position
  logdensityGrad: Array; // Gradient of log density
}
```

#### HMCInfo
```typescript
interface HMCInfo {
  momentum: Array;           // Sampled momentum
  acceptanceProb: Array;     // Metropolis acceptance probability
  isAccepted: Array;         // Whether proposal was accepted
  isDivergent: Array;        // Whether trajectory diverged
  energy: Array;             // Proposal energy
  numIntegrationSteps: number;
}
```

## Memory Management

JAX-JS uses reference counting. Key rules:

1. **Functions consume their inputs.** Use `.ref` to keep an array alive:
   ```typescript
   const y = x.ref.add(1);  // x survives
   const z = x.mul(2);      // x consumed here
   ```

2. **`.js()` consumes the array** (via `dataSync`). Use `.ref` first if you need the array later:
   ```typescript
   const value = arr.ref.js();  // arr survives
   // vs
   const value = arr.js();      // arr consumed
   ```

3. **Dispose arrays when done:**
   ```typescript
   info.momentum.dispose();
   state.position.dispose();
   ```

## Examples

See the `examples/` directory for runnable examples:

- `examples/sample-normal.ts` - Sample from a 1D standard normal distribution
- `examples/sample-normal-2d.ts` - Sample from a 2D standard normal distribution

Run examples with:
```bash
npx tsx examples/sample-normal.ts
npx tsx examples/sample-normal-2d.ts
```

## Development

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Type check
npm run typecheck

# Lint
npm run lint

# Full CI check
npm run ci
```

## Architecture

```
src/
├── integrators/
│   ├── types.ts           # IntegratorState, Integrator types
│   ├── velocity-verlet.ts # Symplectic velocity Verlet integrator
│   └── index.ts
├── metrics/
│   ├── types.ts           # Metric interface
│   ├── gaussian-euclidean.ts # Diagonal mass matrix metric
│   └── index.ts
├── hmc/
│   ├── types.ts           # HMCState, HMCInfo, HMCConfig
│   ├── kernel.ts          # Core HMC algorithm
│   ├── builder.ts         # Fluent builder API
│   └── index.ts
└── index.ts               # Public exports
```

## References

- [The No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) - Hoffman & Gelman
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434) - Betancourt
- [BlackJAX](https://github.com/blackjax-devs/blackjax) - Reference implementation
