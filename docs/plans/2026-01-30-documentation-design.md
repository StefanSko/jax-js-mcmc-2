# Documentation & README Design

**Date:** 2026-01-30
**Goal:** Create a narrative tutorial documentation suite that teaches MCMC from first principles, mapping every concept to the codebase, with interactive JS-based visualizations as teaching instruments.

## Audience

Between "fuzzy practitioner" and "graduate stats." Knows the vocabulary (posterior, Bayes' theorem, MCMC, HMC) but wants the pieces to click together. Comfortable with math notation and formal concepts (ergodicity, detailed balance, Hamiltonian dynamics), but the focus is on *why* and *how they connect*, not on stating definitions.

## Tone

Narrative tutorial. Walks through a story motivated by concrete problems. Rigorous but conversational — like Betancourt's case studies or McElreath's Statistical Rethinking. No forced metaphors.

## README.md

The README is the storefront. ~60 seconds to read. Hook → curiosity → link to tutorial.

```
# jax-js-mcmc

[Screenshot/gif of HMC sampler exploring a 2D banana distribution
 with contour lines and sample trail]

A Bayesian MCMC sampler running in JavaScript, built on JAX-JS.
Watch it explore a distribution:

    npm install && npm run viz

## What you're seeing

The sampler draws from a probability distribution — not by evaluating
a formula, but by exploring the landscape using physics. Each dot is
a proposed position. The chain wanders toward high-probability regions
and stays there.

## Quick example

```ts
import { HMC } from 'jax-js-mcmc'

const logdensity = (x) => x.mul(-0.5).mul(x)  // standard normal
const sampler = HMC(logdensity).stepSize(0.1).numIntegrationSteps(10).build()
// ... run the chain, collect samples
```

## But what's all the buzz about?

You have data. You have a model. Bayes' theorem tells you what to
believe — but the answer is locked behind an integral you can't
compute. MCMC picks that lock.

→ **[Start the tutorial: Why can't I just compute the answer?](docs/tutorial/01-the-problem.md)**

## Install / Run / Test

(standard setup instructions)
```

## Tutorial Series

Problem-first narrative arc. Each chapter ends with a limitation that the next chapter solves. ~800-1200 words per chapter.

### 01-the-problem.md — "You have data. What's the true effect?"

- Concrete problem: you ran an experiment, observed data, want the posterior over parameters
- Bayes' theorem: posterior ∝ likelihood × prior
- The denominator: p(data) = ∫ p(data|θ)p(θ)dθ
- For 1D Gaussian prior + Gaussian likelihood → closed form. Done.
- Now try 10 parameters. Grid with 100 bins per dimension = 10^20 evaluations.
- **Interactive link:** grid approximation in 1D (works), 2D (slow), 3D (hopeless)
- **Ends with:** The grid approach is dead. We need another way.

### 01b-when-posteriors-get-complicated.md — "The prior is simple. The posterior isn't."

- Start with a Gaussian prior on θ. Simple, known, you chose it.
- Multiply by a nonlinear likelihood. The posterior is now... something.
- **Interactive link:** prior × likelihood → posterior shape explorer
  - Dropdown: select different likelihoods
  - Linear-Gaussian → posterior is Gaussian (closed form exists)
  - Logistic regression → posterior is NOT Gaussian
  - Mixture model → posterior is multimodal
- Key distinction: the posterior over parameters is complex, but predictions conditional on a specific θ are simple
- Diagram: posterior(θ|data) → sample θᵢ → predict from p(data|θᵢ). The hard part is the first arrow. That's MCMC's job.
- **Ends with:** The posterior has no formula. We can evaluate it pointwise but can't normalize it. We need samples.

### 02-sampling-replaces-integration.md — "What if we don't need the integral at all?"

- Key insight: if you have samples from the posterior, you can compute any expectation by averaging. E[f(θ)] ≈ (1/N) Σ f(θᵢ)
- Credible intervals, predictions, model comparison — all from samples
- But how do you sample from a distribution you can't normalize?
- Introduce the unnormalized density: we can evaluate p(data|θ)p(θ) for any θ, we just can't integrate it
- **Ends with:** We need an algorithm that samples from an unnormalized distribution.

### 03-random-walk-metropolis.md — "Exploring without a gradient"

- Propose a random step from the current position. Accept if higher density, sometimes accept if lower.
- Detailed balance → the chain's stationary distribution is the posterior
- **Interactive link:** RWM on 2D Gaussian. Slider for step size. Too small = slow exploration. Too large = high rejection.
- Map to code: `src/rwm/kernel.ts` — the accept/reject logic
- **Ends with:** Random proposals don't scale. In high dimensions, most directions move away from the typical set. Acceptance rate collapses.

### 04-hamiltonian-monte-carlo.md — "Using gradients to propose better moves"

- The posterior is a landscape. Treat −log posterior as potential energy. Add momentum as kinetic energy. Now you have a Hamiltonian system.
- Simulate the dynamics → the particle follows contours of the posterior, reaching distant high-probability regions
- **Interactive link:** HMC on same 2D Gaussian. Compare trajectory to RWM. Slider for step size AND integration steps.
- **Interactive link:** Energy conservation — single trajectory, Hamiltonian staying ~constant. Step size slider shows divergence.
- Why it works: proposals are far from the starting point but still high-quality, because physics conserves energy
- **Ends with:** Now you know how it works conceptually. Let's see the code.

### 05-inside-the-code.md — "From math to TypeScript"

- Velocity Verlet integrator: `src/integrators/velocity-verlet.ts` — the leapfrog equations, line by line
- Metric / mass matrix: `src/metrics/gaussian-euclidean.ts` — what "kinetic energy" means in code
- HMC kernel: `src/hmc/kernel.ts` — propose → integrate → accept/reject, mapped to the math
- Builder pattern: `src/hmc/builder.ts` — why immutable config matters
- Reference to tests as executable specification

### 06-memory-and-jax-js.md — "Why this runs in JavaScript at all"

- JAX-JS: autodiff + JIT in the browser
- Reference counting: why `.ref` and `.dispose()` exist
- JIT vs eager: why fusing the step matters for memory
- The tests that keep it honest: refcount tests, physics tests
- Link to memory profiling examples

## Interactive Visualizations

Each interactive is a standalone HTML page. Exists to answer one specific question that's hard to answer with text.

```
examples/visualization/
├── index.html                  # Landing page linking to all interactives
├── grid-approximation/         # Chapter 01: why grids break
│   - 1D grid: slider for bins, posterior converges
│   - 2D grid: computation exploding
│   - 3D grid: counter showing scale (10^6... 10^9...)
│
├── posterior-shapes/           # Chapter 01b: prior × likelihood → complex posterior
│   - Left: prior (Gaussian), Middle: likelihood, Right: posterior
│   - Dropdown: Linear-Gaussian / Logistic / Mixture
│   - Shows posterior warping from simple to complex
│
├── rwm-explorer/               # Chapter 03: what step size does
│   - 2D target (Gaussian or banana, selectable)
│   - RWM chain live, samples accumulating
│   - Slider: step size
│   - Display: acceptance rate, effective sample size, trace plot
│
├── hmc-explorer/               # Chapter 04: trajectories vs random steps
│   - Same 2D targets as RWM for comparison
│   - HMC chain live with trajectory lines
│   - Sliders: step size, integration steps
│   - Toggle: show/hide leapfrog trajectory
│   - Display: acceptance rate, energy error, divergences
│   - Side-by-side mode: RWM vs HMC
│
└── energy-conservation/        # Chapter 04/05: what divergence looks like
    - Single HMC trajectory on contour plot
    - Hamiltonian staying ~constant
    - Slider: step size → too large = energy diverges, rejection
```

## Hosting & Navigation

```
Tutorial markdown    → readable on GitHub (docs/tutorial/*.md)
Interactive viz       → GitHub Pages (examples/visualization/)
README.md            → links to first tutorial chapter

Navigation:
  README → 01 → 01b → 02 → 03 → 04 → 05 → 06
  Each chapter links to next at the bottom.
  Interactive links open in new tab (GitHub Pages).

Build:
  npm run viz         → serves interactives locally (Vite)
  npm run build:viz   → builds static HTML for GitHub Pages
  Tutorial markdown   → no build step
```

## Implementation Phases

### Phase 1: README + first two chapters
- Write README.md (viz screenshot, code example, teaser)
- Write 01-the-problem.md
- Write 01b-when-posteriors-get-complicated.md
- Navigation links between them
- Text only, static images if needed

### Phase 2: RWM + HMC chapters
- Write 03-random-walk-metropolis.md
- Write 04-hamiltonian-monte-carlo.md
- Write 02-sampling-replaces-integration.md (easier to write after 03/04)
- Link to existing `npm run viz` as preview

### Phase 3: Code mapping chapters
- Write 05-inside-the-code.md
- Write 06-memory-and-jax-js.md
- Reference existing tests as executable examples

### Phase 4: Interactive visualizations
- Build posterior-shapes interactive (for 01b)
- Build grid-approximation interactive (for 01)
- Build rwm-explorer interactive (for 03)
- Extend hmc-explorer with side-by-side comparison (for 04)
- Build energy-conservation interactive (for 04/05)
- GitHub Pages deployment
- Link interactives into tutorial chapters
