# Visualization Development

## Verification: Use Debug API, NOT MCP

Do NOT use Playwright MCP tools for browser interaction. Use the existing REST debug API:

- `npm run viz:debug` — starts Vite dev server with console bridge (browser logs → terminal)
- `curl localhost:5173/__debug/state` — get current visualization state
- `curl -X POST localhost:5173/__debug/step` — trigger one step/action
- `curl -X POST localhost:5173/__debug/reset` — reset visualization
- `curl -X POST -d '{"key":"value"}' localhost:5173/__debug/config` — update config
- `curl localhost:5173/__debug/logs?limit=20` — recent console messages

Each visualization must implement `window.__vizDebug = { getState, step, reset, setConfig }`
and run the debug polling loop (copy pattern from `hmc-viz.ts:799-843`).

## JAX-JS Memory Model

JAX-JS uses reference counting. Every array has a refCount.

- `x.ref` — increment refCount (use when you need x again after passing it somewhere)
- `x.dispose()` — decrement refCount (done with x)
- Functions consume their inputs. If you need an input again, use `.ref`

```typescript
// WRONG — double consumption
const y = x.add(1);
const z = x.mul(2);  // x already consumed

// CORRECT
const y = x.ref.add(1);
const z = x.mul(2);  // x consumed here
```

CRITICAL: dispose all JAX-JS arrays when done. Leaking arrays = memory leak.

## Teaching Purpose

The visualizations build a narrative arc: **why do we need MCMC to sample from posteriors?**

1. **grid-approximation/** — You want the posterior. Bayes says posterior ∝ likelihood × prior,
   but you need the normalizing integral. Grid evaluation works in 1D and 2D, but the
   curse of dimensionality makes it hopeless in higher dimensions. *This is why grids fail.*

2. **posterior-shapes/** — Even with a simple Gaussian prior, multiplying by a nonlinear
   likelihood warps the posterior into something with no closed form. Logistic → asymmetric.
   Mixtures → multimodal. You can evaluate the unnormalized density pointwise, but you
   can't integrate it. *This is what makes posteriors hard.*

3. **rwm-explorer/** — First solution: random walk proposals, accept/reject via the
   Metropolis criterion. Samples from the posterior without computing the integral.
   But step size is fragile: too small = slow mixing, too large = high rejection.
   *This is the simplest approach, and its limitations.*

4. **energy-conservation/** — HMC treats −log posterior as potential energy, adds momentum.
   Hamiltonian dynamics conserve total energy → proposals travel far but stay high-quality.
   When the integrator step size is too large, energy conservation breaks → divergence.
   *This is the physics that makes HMC work, and what breaks it.*

5. **hmc-explorer/** — Side-by-side: RWM wanders randomly, HMC follows the geometry.
   Same number of steps, dramatically different exploration. *This is why HMC wins.*

Audience: between "fuzzy practitioner" and "graduate stats." Knows Bayes, MCMC, HMC
as vocabulary, wants the pieces to click together.

## Existing Reusable Components

All imports relative to `examples/visualization/`:

- `renderer.ts` — `CanvasRenderer` class: 2D canvas with zoom/pan, contours, samples, trajectories
- `contour.ts` — `computeDensityGrid`, `computeContourLevels`, `extractContours`
- `distributions.ts` — 7 distributions: gaussian, correlated, banana, bimodal, donut, funnel, squiggle
- `console-bridge.ts` — import at top of .ts file to enable browser→terminal logging
- `shared/trace-plot.ts` — `TracePlot` class: time-series canvas plot

Samplers from `../../src/`:
- `HMC(logdensityFn).stepSize(0.15).numIntegrationSteps(10).build()` → `{ init, step }`
- `RWM(logdensityFn).stepSize(0.2).build()` → `{ init, step }`

## Code Patterns

- Follow `hmc-viz.ts` as the reference implementation
- Each visualization is a standalone HTML page with `<script type="module" src="./viz.ts">`
- No external charting libraries — use Canvas API and existing utilities
- Use pure JS density evaluation for contours (avoid JAX-JS overhead for grid eval)
- Consistent dark theme: background `#1a1a2e`, controls panel `#0f3460`
- Consistent color scheme from `renderer.ts`: green=accepted, red=rejected, blue=current
