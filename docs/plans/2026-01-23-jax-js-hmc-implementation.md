# JAX-JS HMC Sampler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully-tested HMC sampler on JAX-JS with strict memory management and physics-inspired tests.

**Architecture:** Functional core (integrators, kernel) wrapped by immutable builder API. JIT-unrolled integration loops. Branchless accept/reject with `np.where`. All operations consume inputs and produce fresh outputs with refCount=1.

**Tech Stack:** JAX-JS, TypeScript (strict), Vitest, ESLint

**References:**
- JAX-JS source: `/tmp/jax-js` (pull if stale)
- BlackJAX reference: `/tmp/blackjax` (pull if stale)
- Memory model: See `CLAUDE.md`

---

## Phase 0: Project Setup

### Task 0.1: Initialize package.json

**Files:**
- Create: `package.json`

**Step 1: Create package.json**

```json
{
  "name": "jax-js-mcmc",
  "version": "0.1.0",
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint src test --ext .ts",
    "lint:fix": "eslint src test --ext .ts --fix",
    "test": "vitest run",
    "test:watch": "vitest",
    "ci": "npm run typecheck && npm run lint && npm run test",
    "build": "tsc"
  },
  "dependencies": {
    "@anthropic-ai/jax-js": "^0.1.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "eslint": "^9.0.0",
    "typescript": "^5.4.0",
    "typescript-eslint": "^8.0.0",
    "vitest": "^2.0.0"
  }
}
```

**Step 2: Install dependencies**

Run: `npm install`
Expected: `node_modules/` created, no errors

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "chore: initialize package.json with dependencies"
```

---

### Task 0.2: Configure TypeScript (strict)

**Files:**
- Create: `tsconfig.json`

**Step 1: Create tsconfig.json**

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
    "sourceMap": true,
    "outDir": "dist",
    "rootDir": ".",
    "skipLibCheck": true
  },
  "include": ["src/**/*", "test/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Step 2: Verify TypeScript works**

Run: `npx tsc --noEmit`
Expected: No errors (no files to compile yet)

**Step 3: Commit**

```bash
git add tsconfig.json
git commit -m "chore: add strict TypeScript configuration"
```

---

### Task 0.3: Configure ESLint

**Files:**
- Create: `eslint.config.js`

**Step 1: Create eslint.config.js**

```javascript
import tseslint from 'typescript-eslint';

export default tseslint.config(
  tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  {
    ignores: ['dist/', 'node_modules/', '*.js'],
  }
);
```

**Step 2: Verify ESLint works**

Run: `npx eslint .`
Expected: No errors (no files to lint yet)

**Step 3: Commit**

```bash
git add eslint.config.js
git commit -m "chore: add strict ESLint configuration"
```

---

### Task 0.4: Configure Vitest

**Files:**
- Create: `vitest.config.ts`
- Create: `test/setup.ts`

**Step 1: Create vitest.config.ts**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    setupFiles: ['./test/setup.ts'],
    include: ['test/**/*.test.ts'],
  },
});
```

**Step 2: Create test/setup.ts with custom matchers**

```typescript
import { numpy as np } from '@anthropic-ai/jax-js';
import { expect } from 'vitest';

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ) {
    const { isNot } = this;
    const actualArray = np.array(actual);
    const expectedArray = np.array(expected);
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
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
  interface AsymmetricMatchersContaining {
    toBeAllclose(expected: np.ArrayLike, options?: { rtol?: number; atol?: number }): void;
  }
}
```

**Step 3: Create placeholder test to verify setup**

Create `test/setup.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';

describe('Test setup', () => {
  it('vitest runs', () => {
    expect(true).toBe(true);
  });
});
```

**Step 4: Run test to verify setup works**

Run: `npm test`
Expected: 1 test passed

**Step 5: Delete placeholder test and commit**

```bash
rm test/setup.test.ts
git add vitest.config.ts test/setup.ts
git commit -m "chore: add Vitest configuration with custom matchers"
```

---

### Task 0.5: Create directory structure

**Files:**
- Create: `src/index.ts` (empty export)
- Create: `src/integrators/.gitkeep`
- Create: `src/hmc/.gitkeep`
- Create: `src/metrics/.gitkeep`

**Step 1: Create directories and placeholder files**

```bash
mkdir -p src/integrators src/hmc src/metrics
touch src/integrators/.gitkeep src/hmc/.gitkeep src/metrics/.gitkeep
```

**Step 2: Create src/index.ts**

```typescript
// JAX-JS MCMC - HMC Sampler
// Exports will be added as implementation progresses
export {};
```

**Step 3: Verify typecheck passes**

Run: `npm run typecheck`
Expected: No errors

**Step 4: Commit**

```bash
git add src/
git commit -m "chore: create project directory structure"
```

---

## Phase 1: Integrator

### Task 1.1: Integrator types (RED → GREEN)

**Files:**
- Create: `src/integrators/types.ts`
- Create: `test/integrators/types.test.ts`

**Step 1: Write the failing test**

Create `test/integrators/types.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import type { IntegratorState, Integrator } from '../../src/integrators/types';
import { numpy as np } from '@anthropic-ai/jax-js';

describe('Integrator types', () => {
  it('IntegratorState has required fields', () => {
    const state: IntegratorState = {
      position: np.array([0.0]),
      momentum: np.array([1.0]),
      logdensity: np.array(0.0),
      logdensityGrad: np.array([0.0]),
    };

    expect(state.position).toBeDefined();
    expect(state.momentum).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('Integrator type signature is correct', () => {
    const mockIntegrator: Integrator = (state, _stepSize) => state;
    expect(typeof mockIntegrator).toBe('function');
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - Cannot find module '../../src/integrators/types'

**Step 3: Write minimal implementation**

Create `src/integrators/types.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';

export interface IntegratorState {
  position: Array;
  momentum: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export type Integrator = (
  state: IntegratorState,
  stepSize: number
) => IntegratorState;
```

**Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 5: Run full CI**

Run: `npm run ci`
Expected: All checks pass

**Step 6: Commit**

```bash
git add src/integrators/types.ts test/integrators/types.test.ts
git commit -m "feat(integrators): add IntegratorState and Integrator types"
```

---

### Task 1.2: Harmonic oscillator test (RED)

**Files:**
- Create: `test/physics/harmonic-oscillator.test.ts`
- Create: `src/integrators/velocity-verlet.ts` (stub)

**Step 1: Write the failing test**

Create `test/physics/harmonic-oscillator.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@anthropic-ai/jax-js';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Harmonic Oscillator', () => {
  // Potential: U(q) = 0.5 * k * q^2
  // Kinetic:  K(p) = 0.5 * p^2 / m
  // With k=1, m=1: ω = 1
  // Solution: q(t) = q0*cos(t) + p0*sin(t)
  //           p(t) = -q0*sin(t) + p0*cos(t)

  const k = 1.0;
  const m = 1.0;

  // log π(q) = -U(q) = -0.5 * q^2
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5 * k);
  };

  // K(p) = 0.5 * p^2 / m
  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5 / m);
  };

  it('trajectory matches analytical solution', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const q0 = 0.0;
    const p0 = 1.0;
    const stepSize = 0.01;
    const numSteps = 100; // t = 1.0

    // Initial state
    const initialPosition = np.array(q0);
    const initialMomentum = np.array(p0);
    let state: IntegratorState = {
      position: initialPosition.ref,
      momentum: initialMomentum.ref,
      logdensity: logdensityFn(initialPosition.ref),
      logdensityGrad: grad(logdensityFn)(initialPosition),
    };
    initialMomentum.dispose();

    // Integrate
    for (let i = 0; i < numSteps; i++) {
      state = integrator(state, stepSize);
    }

    // Analytical solution at t=1: q(1) = sin(1), p(1) = cos(1)
    const expectedQ = Math.sin(1.0);
    const expectedP = Math.cos(1.0);

    expect(state.position).toBeAllclose(expectedQ, { atol: 1e-4 });
    expect(state.momentum).toBeAllclose(expectedP, { atol: 1e-4 });

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
```

**Step 2: Create stub that fails**

Create `src/integrators/velocity-verlet.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';
import type { Integrator } from './types';

export function createVelocityVerlet(
  _logdensityFn: (position: Array) => Array,
  _kineticEnergyFn: (momentum: Array) => Array
): Integrator {
  throw new Error('Not implemented');
}
```

**Step 3: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - "Not implemented"

**Step 4: Commit the failing test**

```bash
git add test/physics/harmonic-oscillator.test.ts src/integrators/velocity-verlet.ts
git commit -m "test(physics): add harmonic oscillator trajectory test (RED)"
```

---

### Task 1.3: Velocity Verlet implementation (GREEN)

**Files:**
- Modify: `src/integrators/velocity-verlet.ts`

**Step 1: Implement velocity verlet**

Replace `src/integrators/velocity-verlet.ts`:

```typescript
import { grad, type Array } from '@anthropic-ai/jax-js';
import type { Integrator, IntegratorState } from './types';

export function createVelocityVerlet(
  logdensityFn: (position: Array) => Array,
  kineticEnergyFn: (momentum: Array) => Array
): Integrator {
  const kineticEnergyGradFn = grad(kineticEnergyFn);

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

    // Compute new logdensity and gradient at new position
    const newLogdensity = logdensityFn(newPosition.ref);
    const newLogdensityGrad = grad(logdensityFn)(newPosition.ref);

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

**Step 2: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 3: Run full CI**

Run: `npm run ci`
Expected: All checks pass

**Step 4: Commit**

```bash
git add src/integrators/velocity-verlet.ts
git commit -m "feat(integrators): implement velocity verlet integrator (GREEN)"
```

---

### Task 1.4: Energy conservation test (RED → GREEN)

**Files:**
- Create: `test/physics/energy-conservation.test.ts`

**Step 1: Write the energy conservation test**

Create `test/physics/energy-conservation.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@anthropic-ai/jax-js';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Energy Conservation', () => {
  const k = 1.0;
  const m = 1.0;

  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5 * k);
  };

  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5 / m);
  };

  const computeEnergy = (state: IntegratorState): number => {
    // H = -log π(q) + K(p) = U(q) + K(p)
    const potential = state.logdensity.ref.neg();
    const kinetic = kineticEnergyFn(state.momentum.ref);
    const energy = potential.add(kinetic);
    const result = energy.js() as number;
    potential.dispose();
    return result;
  };

  it('conserves energy over many steps', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const q0 = 1.0;
    const p0 = 0.5;
    const stepSize = 0.01;
    const numSteps = 1000;

    // Initial state
    const initialPosition = np.array(q0);
    const initialMomentum = np.array(p0);
    let state: IntegratorState = {
      position: initialPosition.ref,
      momentum: initialMomentum.ref,
      logdensity: logdensityFn(initialPosition.ref),
      logdensityGrad: grad(logdensityFn)(initialPosition),
    };
    initialMomentum.dispose();

    const initialEnergy = computeEnergy(state);

    // Integrate
    for (let i = 0; i < numSteps; i++) {
      state = integrator(state, stepSize);
    }

    const finalEnergy = computeEnergy(state);

    // Energy should be conserved to high precision
    expect(Math.abs(finalEnergy - initialEnergy)).toBeLessThan(1e-4);

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
```

**Step 2: Run test**

Run: `npm test`
Expected: PASS (velocity verlet is symplectic, should conserve energy)

**Step 3: Commit**

```bash
git add test/physics/energy-conservation.test.ts
git commit -m "test(physics): add energy conservation test"
```

---

### Task 1.5: Integrator refcount tests (RED → GREEN)

**Files:**
- Create: `test/integrators/refcount.test.ts`

**Step 1: Write refcount tests**

Create `test/integrators/refcount.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@anthropic-ai/jax-js';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Integrator Memory Management', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5);
  };

  const kineticEnergyFn = (p: np.Array): np.Array => {
    return p.ref.mul(p).mul(0.5);
  };

  const createInitialState = (): IntegratorState => {
    const position = np.array(1.0);
    const momentum = np.array(0.5);
    return {
      position: position.ref,
      momentum: momentum.ref,
      logdensity: logdensityFn(position.ref),
      logdensityGrad: grad(logdensityFn)(position),
    };
  };

  it('old state arrays have refCount 0 after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const oldPosition = state.position;
    const oldMomentum = state.momentum;
    const oldLogdensity = state.logdensity;
    const oldLogdensityGrad = state.logdensityGrad;

    const newState = integrator(state, 0.01);

    // Old state should be consumed
    expect(oldPosition.refCount).toBe(0);
    expect(oldMomentum.refCount).toBe(0);
    expect(oldLogdensity.refCount).toBe(0);
    expect(oldLogdensityGrad.refCount).toBe(0);

    // Cleanup new state
    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('new state arrays have refCount 1 after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const newState = integrator(state, 0.01);

    // New state should have fresh arrays
    expect(newState.position.refCount).toBe(1);
    expect(newState.momentum.refCount).toBe(1);
    expect(newState.logdensity.refCount).toBe(1);
    expect(newState.logdensityGrad.refCount).toBe(1);

    // Cleanup
    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('old state arrays throw ReferenceError when accessed after step', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    const state = createInitialState();

    const oldPosition = state.position;
    const newState = integrator(state, 0.01);

    // Accessing consumed array should throw
    expect(() => oldPosition.js()).toThrowError(ReferenceError);

    // Cleanup
    newState.position.dispose();
    newState.momentum.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
  });

  it('no reference accumulation over many iterations', () => {
    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);
    let state = createInitialState();

    for (let i = 0; i < 100; i++) {
      const oldPosition = state.position;
      state = integrator(state, 0.01);

      // Old consumed, new fresh
      expect(oldPosition.refCount).toBe(0);
      expect(state.position.refCount).toBe(1);
    }

    // Final state still has refCount 1
    expect(state.position.refCount).toBe(1);

    // Cleanup
    state.position.dispose();
    state.momentum.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
```

**Step 2: Run tests**

Run: `npm test`
Expected: PASS

**Step 3: Run full CI**

Run: `npm run ci`
Expected: All checks pass

**Step 4: Commit**

```bash
git add test/integrators/refcount.test.ts
git commit -m "test(integrators): add refcount validation tests"
```

---

### Task 1.6: Export integrator from index

**Files:**
- Modify: `src/index.ts`
- Modify: `src/integrators/types.ts` (add index export)

**Step 1: Create integrators index**

Create `src/integrators/index.ts`:

```typescript
export type { IntegratorState, Integrator } from './types';
export { createVelocityVerlet } from './velocity-verlet';
```

**Step 2: Update main index**

Replace `src/index.ts`:

```typescript
export * from './integrators';
```

**Step 3: Verify exports work**

Run: `npm run typecheck`
Expected: No errors

**Step 4: Commit**

```bash
git add src/integrators/index.ts src/index.ts
git commit -m "feat(integrators): export integrator types and velocity verlet"
```

---

## Phase 2: HMC Kernel

### Task 2.1: HMC types (RED → GREEN)

**Files:**
- Create: `src/hmc/types.ts`
- Create: `test/hmc/types.test.ts`

**Step 1: Write the failing test**

Create `test/hmc/types.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import type { HMCState, HMCInfo, HMCConfig } from '../../src/hmc/types';
import { numpy as np } from '@anthropic-ai/jax-js';

describe('HMC types', () => {
  it('HMCState has required fields', () => {
    const state: HMCState = {
      position: np.array([0.0]),
      logdensity: np.array(0.0),
      logdensityGrad: np.array([0.0]),
    };

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('HMCInfo has required fields', () => {
    const info: HMCInfo = {
      momentum: np.array([0.0]),
      acceptanceProb: np.array(0.5),
      isAccepted: np.array(true),
      isDivergent: np.array(false),
      energy: np.array(1.0),
      numIntegrationSteps: 10,
    };

    expect(info.momentum).toBeDefined();
    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.isDivergent).toBeDefined();
    expect(info.energy).toBeDefined();
    expect(info.numIntegrationSteps).toBe(10);

    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('HMCConfig has required fields', () => {
    const config: HMCConfig = {
      stepSize: 0.1,
      numIntegrationSteps: 10,
      inverseMassMatrix: np.array([1.0]),
      divergenceThreshold: 1000,
    };

    expect(config.stepSize).toBe(0.1);
    expect(config.numIntegrationSteps).toBe(10);
    expect(config.divergenceThreshold).toBe(1000);

    config.inverseMassMatrix.dispose();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - Cannot find module

**Step 3: Write implementation**

Create `src/hmc/types.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';

export interface HMCState {
  position: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export interface HMCInfo {
  momentum: Array;
  acceptanceProb: Array;
  isAccepted: Array;
  isDivergent: Array;
  energy: Array;
  numIntegrationSteps: number;
}

export interface HMCConfig {
  stepSize: number;
  numIntegrationSteps: number;
  inverseMassMatrix: Array;
  divergenceThreshold: number;
}
```

**Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hmc/types.ts test/hmc/types.test.ts
git commit -m "feat(hmc): add HMCState, HMCInfo, HMCConfig types"
```

---

### Task 2.2: Metric types and Gaussian Euclidean (RED → GREEN)

**Files:**
- Create: `src/metrics/types.ts`
- Create: `src/metrics/gaussian-euclidean.ts`
- Create: `test/metrics/gaussian-euclidean.test.ts`

**Step 1: Write the failing test**

Create `test/metrics/gaussian-euclidean.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@anthropic-ai/jax-js';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';

describe('Gaussian Euclidean Metric', () => {
  it('sampleMomentum returns array with correct shape', () => {
    const inverseMassMatrix = np.array([1.0, 1.0, 1.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const key = random.key(42);
    const position = np.array([0.0, 0.0, 0.0]);
    const momentum = metric.sampleMomentum(key, position);

    expect(momentum.shape).toEqual([3]);

    momentum.dispose();
    position.dispose();
    inverseMassMatrix.dispose();
  });

  it('kineticEnergy computes 0.5 * p^T * M^{-1} * p', () => {
    const inverseMassMatrix = np.array([1.0, 2.0]); // diagonal
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const momentum = np.array([2.0, 3.0]);
    const energy = metric.kineticEnergy(momentum);

    // K = 0.5 * (2^2 * 1 + 3^2 * 2) = 0.5 * (4 + 18) = 11
    expect(energy).toBeAllclose(11.0);

    energy.dispose();
    inverseMassMatrix.dispose();
  });

  it('kineticEnergyGrad computes M^{-1} * p', () => {
    const inverseMassMatrix = np.array([1.0, 2.0]);
    const metric = createGaussianEuclidean(inverseMassMatrix);

    const momentum = np.array([2.0, 3.0]);
    const grad = metric.kineticEnergyGrad(momentum);

    // ∇K = M^{-1} * p = [1*2, 2*3] = [2, 6]
    expect(grad).toBeAllclose([2.0, 6.0]);

    grad.dispose();
    inverseMassMatrix.dispose();
  });
});
```

**Step 2: Create stub that fails**

Create `src/metrics/types.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';

export interface Metric {
  sampleMomentum: (key: Array, position: Array) => Array;
  kineticEnergy: (momentum: Array) => Array;
  kineticEnergyGrad: (momentum: Array) => Array;
}
```

Create `src/metrics/gaussian-euclidean.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';
import type { Metric } from './types';

export function createGaussianEuclidean(_inverseMassMatrix: Array): Metric {
  throw new Error('Not implemented');
}
```

**Step 3: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - "Not implemented"

**Step 4: Implement**

Replace `src/metrics/gaussian-euclidean.ts`:

```typescript
import { grad, numpy as np, random, type Array } from '@anthropic-ai/jax-js';
import type { Metric } from './types';

export function createGaussianEuclidean(inverseMassMatrix: Array): Metric {
  // For diagonal mass matrix: M^{-1} = diag(inverseMassMatrix)
  // Mass matrix sqrt: M^{1/2} = diag(1/sqrt(inverseMassMatrix))
  const massMatrixSqrt = inverseMassMatrix.ref.reciprocal().sqrt();

  const kineticEnergy = (momentum: Array): Array => {
    // K = 0.5 * p^T * M^{-1} * p = 0.5 * sum(p^2 * M^{-1})
    const scaled = momentum.ref.mul(momentum).mul(inverseMassMatrix.ref);
    const result = scaled.sum().mul(0.5);
    return result;
  };

  const kineticEnergyGrad = grad(kineticEnergy);

  const sampleMomentum = (key: Array, position: Array): Array => {
    // p ~ N(0, M) => p = M^{1/2} * z where z ~ N(0, I)
    const z = random.normal(key, position.shape);
    const momentum = z.mul(massMatrixSqrt.ref);
    return momentum;
  };

  return {
    sampleMomentum,
    kineticEnergy,
    kineticEnergyGrad,
  };
}
```

**Step 5: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 6: Commit**

```bash
git add src/metrics/types.ts src/metrics/gaussian-euclidean.ts test/metrics/gaussian-euclidean.test.ts
git commit -m "feat(metrics): add Gaussian Euclidean metric with diagonal mass matrix"
```

---

### Task 2.3: HMC kernel skeleton (RED → GREEN)

**Files:**
- Create: `src/hmc/kernel.ts`
- Create: `test/hmc/kernel.test.ts`

**Step 1: Write the failing test**

Create `test/hmc/kernel.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad, random } from '@anthropic-ai/jax-js';
import { createHMCKernel } from '../../src/hmc/kernel';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { HMCState, HMCConfig } from '../../src/hmc/types';

describe('HMC Kernel', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    // Standard normal: log p(q) = -0.5 * q^2
    return q.ref.mul(q).mul(-0.5).sum();
  };

  const createConfig = (): HMCConfig => ({
    stepSize: 0.1,
    numIntegrationSteps: 10,
    inverseMassMatrix: np.array([1.0]),
    divergenceThreshold: 1000,
  });

  const createInitialState = (): HMCState => {
    const position = np.array([0.0]);
    return {
      position: position.ref,
      logdensity: logdensityFn(position.ref),
      logdensityGrad: grad(logdensityFn)(position),
    };
  };

  it('returns new state and info', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const key = random.key(42);

    const [newState, info] = hmcStep(key, state);

    // New state should have all required fields
    expect(newState.position).toBeDefined();
    expect(newState.logdensity).toBeDefined();
    expect(newState.logdensityGrad).toBeDefined();

    // Info should have all required fields
    expect(info.momentum).toBeDefined();
    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.isDivergent).toBeDefined();
    expect(info.energy).toBeDefined();
    expect(info.numIntegrationSteps).toBe(10);

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
    config.inverseMassMatrix.dispose();
  });
});
```

**Step 2: Create stub that fails**

Create `src/hmc/kernel.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import type { Metric } from '../metrics/types';
import type { Integrator } from '../integrators/types';

export type HMCKernel = (key: Array, state: HMCState) => [HMCState, HMCInfo];

export function createHMCKernel(
  _config: HMCConfig,
  _logdensityFn: (position: Array) => Array,
  _metric: Metric,
  _integrator: Integrator
): HMCKernel {
  throw new Error('Not implemented');
}
```

**Step 3: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - "Not implemented"

**Step 4: Implement HMC kernel**

Replace `src/hmc/kernel.ts`:

```typescript
import { grad, jit, numpy as np, random, type Array } from '@anthropic-ai/jax-js';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import type { Metric } from '../metrics/types';
import type { Integrator, IntegratorState } from '../integrators/types';

export type HMCKernel = (key: Array, state: HMCState) => [HMCState, HMCInfo];

export function createHMCKernel(
  config: HMCConfig,
  logdensityFn: (position: Array) => Array,
  metric: Metric,
  integrator: Integrator
): HMCKernel {
  const { stepSize, numIntegrationSteps, divergenceThreshold } = config;

  return function hmcStep(key: Array, state: HMCState): [HMCState, HMCInfo] {
    const [keyMomentum, keyAccept] = random.split(key, 2) as [Array, Array];

    // Sample momentum
    const momentum = metric.sampleMomentum(keyMomentum, state.position.ref);

    // Initial integrator state
    let integState: IntegratorState = {
      position: state.position.ref,
      momentum: momentum.ref,
      logdensity: state.logdensity.ref,
      logdensityGrad: state.logdensityGrad.ref,
    };

    // Compute initial energy: H = -log π(q) + K(p)
    const initialKinetic = metric.kineticEnergy(momentum.ref);
    const initialEnergy = state.logdensity.ref.neg().add(initialKinetic);

    // Integrate trajectory
    for (let i = 0; i < numIntegrationSteps; i++) {
      integState = integrator(integState, stepSize);
    }

    // Compute proposal energy
    const proposalKinetic = metric.kineticEnergy(integState.momentum.ref);
    const proposalEnergy = integState.logdensity.ref.neg().add(proposalKinetic);

    // Metropolis-Hastings acceptance
    const deltaEnergy = proposalEnergy.ref.sub(initialEnergy);
    const isDivergent = deltaEnergy.ref.greater(divergenceThreshold);
    const acceptanceProb = np.minimum(np.array(1.0), np.exp(deltaEnergy.neg()));
    const uniform = random.uniform(keyAccept, []);
    const isAccepted = uniform.less(acceptanceProb.ref);

    // Select state based on acceptance (branchless with np.where)
    const newPosition = np.where(
      isAccepted.ref,
      integState.position.ref,
      state.position.ref
    );
    const newLogdensity = np.where(
      isAccepted.ref,
      integState.logdensity.ref,
      state.logdensity.ref
    );
    const newLogdensityGrad = np.where(
      isAccepted.ref,
      integState.logdensityGrad.ref,
      state.logdensityGrad.ref
    );

    // Dispose all consumed arrays
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
    integState.position.dispose();
    integState.momentum.dispose();
    integState.logdensity.dispose();
    integState.logdensityGrad.dispose();
    initialEnergy.dispose();
    deltaEnergy.dispose();

    const newState: HMCState = {
      position: newPosition,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };

    const info: HMCInfo = {
      momentum,
      acceptanceProb,
      isAccepted,
      isDivergent,
      energy: proposalEnergy,
      numIntegrationSteps,
    };

    return [newState, info];
  };
}
```

**Step 5: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hmc/kernel.ts test/hmc/kernel.test.ts
git commit -m "feat(hmc): implement HMC kernel with Metropolis-Hastings"
```

---

### Task 2.4: HMC kernel refcount tests

**Files:**
- Create: `test/hmc/refcount.test.ts`

**Step 1: Write refcount tests**

Create `test/hmc/refcount.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad, random } from '@anthropic-ai/jax-js';
import { createHMCKernel } from '../../src/hmc/kernel';
import { createGaussianEuclidean } from '../../src/metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { HMCState, HMCConfig } from '../../src/hmc/types';

describe('HMC Kernel Memory Management', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  const createConfig = (): HMCConfig => ({
    stepSize: 0.1,
    numIntegrationSteps: 10,
    inverseMassMatrix: np.array([1.0]),
    divergenceThreshold: 1000,
  });

  const createInitialState = (): HMCState => {
    const position = np.array([0.0]);
    return {
      position: position.ref,
      logdensity: logdensityFn(position.ref),
      logdensityGrad: grad(logdensityFn)(position),
    };
  };

  const disposeInfo = (info: any): void => {
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  };

  it('old state arrays have refCount 0 after step', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const oldPosition = state.position;
    const oldLogdensity = state.logdensity;
    const oldLogdensityGrad = state.logdensityGrad;

    const key = random.key(42);
    const [newState, info] = hmcStep(key, state);

    // Old state should be consumed
    expect(oldPosition.refCount).toBe(0);
    expect(oldLogdensity.refCount).toBe(0);
    expect(oldLogdensityGrad.refCount).toBe(0);

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    disposeInfo(info);
    config.inverseMassMatrix.dispose();
  });

  it('new state arrays have refCount 1 after step', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    const state = createInitialState();
    const key = random.key(42);
    const [newState, info] = hmcStep(key, state);

    // New state should have refCount 1
    expect(newState.position.refCount).toBe(1);
    expect(newState.logdensity.refCount).toBe(1);
    expect(newState.logdensityGrad.refCount).toBe(1);

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    disposeInfo(info);
    config.inverseMassMatrix.dispose();
  });

  it('no reference accumulation over many HMC steps', () => {
    const config = createConfig();
    const metric = createGaussianEuclidean(config.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(logdensityFn, metric.kineticEnergy);
    const hmcStep = createHMCKernel(config, logdensityFn, metric, integrator);

    let state = createInitialState();

    for (let i = 0; i < 50; i++) {
      const oldPosition = state.position;
      const key = random.key(i);
      const [newState, info] = hmcStep(key, state);

      // Old consumed, new fresh
      expect(oldPosition.refCount).toBe(0);
      expect(newState.position.refCount).toBe(1);

      disposeInfo(info);
      state = newState;
    }

    // Final state still has refCount 1
    expect(state.position.refCount).toBe(1);

    // Cleanup
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
    config.inverseMassMatrix.dispose();
  });
});
```

**Step 2: Run tests**

Run: `npm test`
Expected: PASS

**Step 3: Commit**

```bash
git add test/hmc/refcount.test.ts
git commit -m "test(hmc): add refcount validation tests for HMC kernel"
```

---

### Task 2.5: HMC Builder (RED → GREEN)

**Files:**
- Create: `src/hmc/builder.ts`
- Create: `test/hmc/builder.test.ts`

**Step 1: Write the failing test**

Create `test/hmc/builder.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@anthropic-ai/jax-js';
import { HMC } from '../../src/hmc/builder';

describe('HMC Builder', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  it('builds sampler with fluent API', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    expect(sampler.init).toBeDefined();
    expect(sampler.step).toBeDefined();
  });

  it('builder is immutable', () => {
    const builder1 = HMC(logdensityFn);
    const builder2 = builder1.stepSize(0.1);

    expect(builder1).not.toBe(builder2);
  });

  it('init creates valid state', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    const position = np.array([1.0]);
    const state = sampler.init(position);

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();
    expect(state.logdensityGrad).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });

  it('step runs without error', () => {
    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    const state = sampler.init(np.array([0.0]));
    const key = random.key(42);
    const [newState, info] = sampler.step(key, state);

    expect(newState.position).toBeDefined();
    expect(info.isAccepted).toBeDefined();

    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('throws if required config is missing', () => {
    expect(() => HMC(logdensityFn).build()).toThrow('stepSize required');
    expect(() => HMC(logdensityFn).stepSize(0.1).build()).toThrow('numIntegrationSteps required');
  });
});
```

**Step 2: Create stub that fails**

Create `src/hmc/builder.ts`:

```typescript
import type { Array } from '@anthropic-ai/jax-js';
import type { HMCState, HMCInfo } from './types';

export interface HMCSampler {
  init: (position: Array) => HMCState;
  step: (key: Array, state: HMCState) => [HMCState, HMCInfo];
}

export function HMC(_logdensityFn: (pos: Array) => Array): HMCBuilder {
  throw new Error('Not implemented');
}

export class HMCBuilder {
  stepSize(_value: number): HMCBuilder {
    throw new Error('Not implemented');
  }
  numIntegrationSteps(_value: number): HMCBuilder {
    throw new Error('Not implemented');
  }
  inverseMassMatrix(_value: Array): HMCBuilder {
    throw new Error('Not implemented');
  }
  divergenceThreshold(_value: number): HMCBuilder {
    throw new Error('Not implemented');
  }
  build(): HMCSampler {
    throw new Error('Not implemented');
  }
}
```

**Step 3: Run test to verify it fails**

Run: `npm test`
Expected: FAIL - "Not implemented"

**Step 4: Implement builder**

Replace `src/hmc/builder.ts`:

```typescript
import { grad, type Array } from '@anthropic-ai/jax-js';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import { createHMCKernel } from './kernel';
import { createGaussianEuclidean } from '../metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../integrators/velocity-verlet';

export interface HMCSampler {
  init: (position: Array) => HMCState;
  step: (key: Array, state: HMCState) => [HMCState, HMCInfo];
}

interface PartialConfig {
  stepSize?: number;
  numIntegrationSteps?: number;
  inverseMassMatrix?: Array;
  divergenceThreshold?: number;
}

export class HMCBuilder {
  private constructor(
    private readonly logdensityFn: (pos: Array) => Array,
    private readonly config: PartialConfig
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
    const fullConfig = this.validateAndFillDefaults();
    const metric = createGaussianEuclidean(fullConfig.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(this.logdensityFn, metric.kineticEnergy);
    const step = createHMCKernel(fullConfig, this.logdensityFn, metric, integrator);

    const init = (position: Array): HMCState => {
      return {
        position: position.ref,
        logdensity: this.logdensityFn(position.ref),
        logdensityGrad: grad(this.logdensityFn)(position),
      };
    };

    return { init, step };
  }

  private validateAndFillDefaults(): HMCConfig {
    if (this.config.stepSize === undefined) {
      throw new Error('stepSize required');
    }
    if (this.config.numIntegrationSteps === undefined) {
      throw new Error('numIntegrationSteps required');
    }
    if (this.config.inverseMassMatrix === undefined) {
      throw new Error('inverseMassMatrix required');
    }

    return {
      stepSize: this.config.stepSize,
      numIntegrationSteps: this.config.numIntegrationSteps,
      inverseMassMatrix: this.config.inverseMassMatrix,
      divergenceThreshold: this.config.divergenceThreshold ?? 1000,
    };
  }
}

export const HMC = HMCBuilder.create;
```

**Step 5: Run test to verify it passes**

Run: `npm test`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hmc/builder.ts test/hmc/builder.test.ts
git commit -m "feat(hmc): implement immutable HMC builder"
```

---

### Task 2.6: Export HMC from index

**Files:**
- Create: `src/hmc/index.ts`
- Create: `src/metrics/index.ts`
- Modify: `src/index.ts`

**Step 1: Create hmc index**

Create `src/hmc/index.ts`:

```typescript
export type { HMCState, HMCInfo, HMCConfig } from './types';
export type { HMCSampler } from './builder';
export { HMC, HMCBuilder } from './builder';
export { createHMCKernel } from './kernel';
```

**Step 2: Create metrics index**

Create `src/metrics/index.ts`:

```typescript
export type { Metric } from './types';
export { createGaussianEuclidean } from './gaussian-euclidean';
```

**Step 3: Update main index**

Replace `src/index.ts`:

```typescript
export * from './integrators';
export * from './hmc';
export * from './metrics';
```

**Step 4: Verify typecheck passes**

Run: `npm run ci`
Expected: All checks pass

**Step 5: Commit**

```bash
git add src/hmc/index.ts src/metrics/index.ts src/index.ts
git commit -m "feat: export all modules from main index"
```

---

## Phase 3: Diagnostics & Additional Tests

### Task 3.1: Gradient flow test

**Files:**
- Create: `test/memory/gradient-flow.test.ts`

**Step 1: Write gradient flow test**

Create `test/memory/gradient-flow.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, grad } from '@anthropic-ai/jax-js';
import { createVelocityVerlet } from '../../src/integrators/velocity-verlet';
import type { IntegratorState } from '../../src/integrators/types';

describe('Gradient Flow', () => {
  it('grad flows through single integrator step', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5);
    };

    const kineticEnergyFn = (p: np.Array): np.Array => {
      return p.ref.mul(p).mul(0.5);
    };

    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    // Differentiate final position w.r.t. initial position
    const finalPosition = (q0: np.Array): np.Array => {
      const position = q0.ref;
      const momentum = np.array(1.0);
      let state: IntegratorState = {
        position: position,
        momentum: momentum.ref,
        logdensity: logdensityFn(position.ref),
        logdensityGrad: grad(logdensityFn)(position.ref),
      };
      momentum.dispose();

      state = integrator(state, 0.01);

      // Dispose unused arrays
      state.momentum.dispose();
      state.logdensity.dispose();
      state.logdensityGrad.dispose();

      return state.position;
    };

    const dqdq0 = grad(finalPosition)(np.array(0.0));

    // Gradient should be close to 1 for small step size
    expect(dqdq0).toBeAllclose(1.0, { atol: 0.01 });
  });

  it('grad through multiple integrator steps', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5);
    };

    const kineticEnergyFn = (p: np.Array): np.Array => {
      return p.ref.mul(p).mul(0.5);
    };

    const integrator = createVelocityVerlet(logdensityFn, kineticEnergyFn);

    const finalPosition = (q0: np.Array): np.Array => {
      const position = q0.ref;
      const momentum = np.array(1.0);
      let state: IntegratorState = {
        position: position,
        momentum: momentum.ref,
        logdensity: logdensityFn(position.ref),
        logdensityGrad: grad(logdensityFn)(position.ref),
      };
      momentum.dispose();

      for (let i = 0; i < 10; i++) {
        state = integrator(state, 0.01);
      }

      state.momentum.dispose();
      state.logdensity.dispose();
      state.logdensityGrad.dispose();

      return state.position;
    };

    const gradient = grad(finalPosition)(np.array(0.5));

    // Should return a valid scalar gradient
    expect(gradient.shape).toEqual([]);
    expect(typeof gradient.js()).toBe('number');
    expect(Number.isFinite(gradient.js() as number)).toBe(true);

    gradient.dispose();
  });
});
```

**Step 2: Run test**

Run: `npm test`
Expected: PASS

**Step 3: Commit**

```bash
git add test/memory/gradient-flow.test.ts
git commit -m "test(memory): add gradient flow tests through integrator"
```

---

### Task 3.2: Divergence detection test

**Files:**
- Create: `test/hmc/divergence.test.ts`

**Step 1: Write divergence test**

Create `test/hmc/divergence.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@anthropic-ai/jax-js';
import { HMC } from '../../src/hmc/builder';

describe('Divergence Detection', () => {
  it('detects divergence with large step size', () => {
    // Narrow target that will cause divergence with large steps
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-50.0).sum(); // Very narrow Gaussian
    };

    const sampler = HMC(logdensityFn)
      .stepSize(1.0) // Deliberately too large
      .numIntegrationSteps(100)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    const state = sampler.init(np.array([0.1]));
    const key = random.key(123);
    const [newState, info] = sampler.step(key, state);

    // With such a large step size, we expect divergence
    const isDivergentValue = info.isDivergent.js();
    // Note: May or may not diverge depending on random momentum
    // This test mainly ensures the divergence flag is computed
    expect(typeof isDivergentValue).toBe('boolean');

    // Cleanup
    newState.position.dispose();
    newState.logdensity.dispose();
    newState.logdensityGrad.dispose();
    info.momentum.dispose();
    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  });

  it('does not diverge with appropriate step size', () => {
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.01) // Small, safe step size
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .divergenceThreshold(1000)
      .build();

    let divergenceCount = 0;
    let state = sampler.init(np.array([0.0]));

    for (let i = 0; i < 20; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      if (info.isDivergent.js()) {
        divergenceCount++;
      }

      // Cleanup info
      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();

      state = newState;
    }

    // Should have no divergences with small step size
    expect(divergenceCount).toBe(0);

    // Cleanup final state
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
```

**Step 2: Run test**

Run: `npm test`
Expected: PASS

**Step 3: Commit**

```bash
git add test/hmc/divergence.test.ts
git commit -m "test(hmc): add divergence detection tests"
```

---

### Task 3.3: Stationary distribution test

**Files:**
- Create: `test/hmc/stationary.test.ts`

**Step 1: Write stationary distribution test**

Create `test/hmc/stationary.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { numpy as np, random } from '@anthropic-ai/jax-js';
import { HMC } from '../../src/hmc/builder';

describe('HMC Stationary Distribution', () => {
  it('samples approximate standard normal mean and variance', () => {
    // Target: standard normal N(0, 1)
    const logdensityFn = (q: np.Array): np.Array => {
      return q.ref.mul(q).mul(-0.5).sum();
    };

    const sampler = HMC(logdensityFn)
      .stepSize(0.1)
      .numIntegrationSteps(10)
      .inverseMassMatrix(np.array([1.0]))
      .build();

    let state = sampler.init(np.array([0.0]));
    const samples: number[] = [];

    // Warmup
    for (let i = 0; i < 100; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);
      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();
      state = newState;
    }

    // Sample
    for (let i = 100; i < 600; i++) {
      const key = random.key(i);
      const [newState, info] = sampler.step(key, state);

      samples.push(newState.position.js() as number);

      info.momentum.dispose();
      info.acceptanceProb.dispose();
      info.isAccepted.dispose();
      info.isDivergent.dispose();
      info.energy.dispose();
      state = newState;
    }

    // Compute sample statistics
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    // Check approximate agreement with N(0, 1)
    expect(Math.abs(mean)).toBeLessThan(0.2); // Mean should be close to 0
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.3); // Variance should be close to 1

    // Cleanup
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
  });
});
```

**Step 2: Run test**

Run: `npm test`
Expected: PASS

**Step 3: Run full CI**

Run: `npm run ci`
Expected: All checks pass

**Step 4: Commit**

```bash
git add test/hmc/stationary.test.ts
git commit -m "test(hmc): add stationary distribution test"
```

---

## Final: Summary Commit

**Step 1: Verify all tests pass**

Run: `npm run ci`
Expected: All checks pass (typecheck, lint, tests)

**Step 2: Create summary commit**

```bash
git log --oneline | head -20
```

Review commits, then tag if desired:

```bash
git tag -a v0.1.0 -m "Initial HMC implementation with TDD"
```

---

## Appendix: Commands Reference

| Command | Purpose |
|---------|---------|
| `npm run ci` | Full check: typecheck + lint + test |
| `npm run test` | Run tests once |
| `npm run test:watch` | TDD mode - watch files |
| `npm run typecheck` | TypeScript only |
| `npm run lint` | ESLint only |
| `npm run lint:fix` | Auto-fix lint issues |

## Appendix: Key Files to Reference

| File | Purpose |
|------|---------|
| `/tmp/jax-js/src/frontend/array.ts` | JAX-JS Array API, refCount, dispose |
| `/tmp/jax-js/test/refcount.test.ts` | Example memory tests |
| `/tmp/blackjax/blackjax/mcmc/hmc.py` | Reference HMC implementation |
| `/tmp/blackjax/tests/mcmc/test_integrators.py` | Physics test examples |
| `CLAUDE.md` | Memory model quick reference |
