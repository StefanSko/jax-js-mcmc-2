# HMC/RWM Visualization

Interactive browser-based visualization of Hamiltonian Monte Carlo (HMC) and Random Walk Metropolis (RWM) sampling using JAX-JS.

## Quick Start

From the project root:

```bash
# Development with hot-reload
npm run viz

# Or build as static files
npm run viz:build
npm run viz:preview
```

Then open http://localhost:5173 (dev) or http://localhost:4173 (preview) in your browser.

## Features

### WebGPU Acceleration

The visualization automatically detects and uses WebGPU if available, falling back to WASM otherwise. WebGPU provides significant speedup for the underlying JAX-JS computations.

### Distributions

Seven distributions are available, each demonstrating different HMC behaviors:

| Distribution | Description | Challenge |
|-------------|-------------|-----------|
| **2D Gaussian** | Standard isotropic normal | Simple baseline |
| **Correlated Gaussian** | 2D Gaussian with ρ=0.9 | Correlation handling |
| **Banana** | Rosenbrock-like curved posterior | Non-linear geometry |
| **Bimodal** | Two separated Gaussian modes | Mode jumping |
| **Donut** | Ring-shaped density at r=2.5 | Curved manifold |
| **Funnel** | Neal's funnel with varying scales | Scale variation |
| **Squiggle** | Sinusoidal ridge y=1.5·sin(x) | Winding path |

### True Parameters Display

Toggle "Show True Parameters" to display:
- **Star markers (★)** - Distribution modes (density maxima)
- **Diamond markers (◆)** - Distribution mean (if different from mode)

This helps verify that MCMC samples are concentrating around the correct regions.

### Controls

| Control | Description |
|---------|-------------|
| **Algorithm** | Choose between HMC (gradient-based) and RWM (random walk) |
| **Step Size (ε)** | Leapfrog integration step size. Larger = more aggressive proposals |
| **Integration Steps (L)** | Number of leapfrog steps per HMC iteration (hidden for RWM) |
| **Animation Speed** | Milliseconds between samples |
| **Zoom** | Buttons or mouse wheel to zoom in/out on the canvas |

### Visualization Legend

- **Green dots** - Accepted samples
- **Red dots** - Rejected samples
- **Yellow dots** - Divergent transitions (HMC only)
- **Blue dot** - Current sampler position
- **Gray contours** - Log-density contour lines
- **Pink stars** - True modes
- **Purple diamonds** - True mean

## HMC Physics

### Acceptance Rate

HMC acceptance is determined by energy conservation, not proximity to the mode. The algorithm:
1. Samples random momentum
2. Simulates Hamiltonian dynamics for L steps
3. Accepts/rejects based on total energy change

High acceptance (>90%) with small step sizes is expected - the integrator accurately preserves energy. This doesn't mean the sampler is stuck; it's taking small but valid steps.

### Step Size Effects

| Step Size | Behavior |
|-----------|----------|
| Too small | 100% acceptance but slow exploration |
| Optimal | 65-80% acceptance, efficient mixing |
| Too large | Low acceptance, divergences |

### Divergences

Divergent transitions (yellow) indicate numerical instability from:
- Step size too large for local curvature
- Extreme scale variation (like Neal's Funnel)
- Approaching distribution boundaries

## RWM Notes

RWM proposes `q' = q + ε * noise` without gradients or momentum. It typically mixes more slowly than HMC, so expect lower acceptance rates and more random-walk behavior for larger step sizes.

## Debug API (Development)

The dev server exposes REST endpoints for programmatic control and agentic debugging.

### Quick Test

```bash
npm run viz &
sleep 5
curl http://localhost:5173/__debug/state
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/__debug/state` | Full sampler state (algorithm, position, stats, config, recent samples) |
| POST | `/__debug/step` | Execute one step, returns acceptance info |
| POST | `/__debug/reset` | Reset to initial state |
| POST | `/__debug/config` | Update algorithm/distribution/parameters |
| GET | `/__debug/logs?limit=N` | Console messages (requires `CONSOLE_BRIDGE=1`) |

### Examples

```bash
# Get current state
curl http://localhost:5173/__debug/state

# Execute single step
curl -X POST http://localhost:5173/__debug/step

# Reset sampler
curl -X POST http://localhost:5173/__debug/reset

# Switch to RWM algorithm
curl -X POST -H "Content-Type: application/json" \
  -d '{"algorithm":"rwm"}' \
  http://localhost:5173/__debug/config

# Change distribution and step size
curl -X POST -H "Content-Type: application/json" \
  -d '{"distribution":"banana","stepSize":0.1}' \
  http://localhost:5173/__debug/config

# Full config update
curl -X POST -H "Content-Type: application/json" \
  -d '{"algorithm":"hmc","stepSize":0.15,"numSteps":10,"distribution":"funnel"}' \
  http://localhost:5173/__debug/config
```

### Response Formats

**GET /__debug/state**
```json
{
  "algorithm": "hmc",
  "distribution": "2D Gaussian",
  "position": [0.5, -0.3],
  "stepCount": 42,
  "acceptedCount": 38,
  "divergentCount": 0,
  "acceptanceRate": 0.905,
  "config": {"stepSize": 0.15, "numIntegrationSteps": 10},
  "recentSamples": [{"x": 0.5, "y": -0.3, "accepted": true, "divergent": false}]
}
```

**POST /__debug/step**
```json
{
  "accepted": true,
  "acceptanceProb": 0.987,
  "position": [0.72, -0.15],
  "energy": 1.234,
  "isDivergent": false
}
```
Note: `energy` and `isDivergent` only present for HMC.

### Config Options

| Option | Values |
|--------|--------|
| `algorithm` | `"hmc"`, `"rwm"` |
| `stepSize` | float (e.g., `0.1`) |
| `numSteps` | integer (HMC only, e.g., `10`) |
| `distribution` | `"gaussian"`, `"correlated"`, `"banana"`, `"bimodal"`, `"donut"`, `"funnel"`, `"squiggle"` |

### Run Multiple Steps

```bash
for i in {1..10}; do
  curl -s -X POST http://localhost:5173/__debug/step | jq -c '{accepted, acceptanceProb}'
done
```

### Browser Console API

Also available in browser devtools:

```javascript
__hmcDebug.getState()
__hmcDebug.step()
__hmcDebug.reset()
__hmcDebug.setConfig({ algorithm: 'rwm', stepSize: 0.3 })
```

### Architecture

The debug API uses a command queue pattern for browser-server communication:

```
Terminal ──curl──▶ Vite Server ◀──poll/result──▶ Browser
                  (debug-api.ts)                 (hmc-viz.ts)
```

1. curl request queues command on server
2. Browser polls `/__debug/poll` every 100ms
3. Browser executes, posts result to `/__debug/result`
4. Server returns result to curl (5s timeout)

### Legacy API

The older `/__api/*` endpoints are still available:

```bash
curl http://localhost:5173/__api/status
curl -X POST http://localhost:5173/__api/play
curl -X POST http://localhost:5173/__api/pause
curl -X POST http://localhost:5173/__api/step
curl -X POST http://localhost:5173/__api/reset
curl http://localhost:5173/__api/distributions
curl -X POST http://localhost:5173/__api/distribution/banana
curl -X POST http://localhost:5173/__api/stepsize/0.3
curl -X POST http://localhost:5173/__api/numsteps/20
```

## Static Build

Build for static deployment (no server required):

```bash
npm run viz:build
```

Output is in `dist/visualization/`. Deploy to any static host (GitHub Pages, Netlify, etc.).

The static build works identically to the dev version - all computation happens client-side in the browser.

## Technical Notes

- JAX-JS has a warmup cost for autodiff compilation on first sample
- Samples are limited to 500 to prevent memory issues
- Contours use pure JavaScript (marching squares) for speed
- WebGPU provides ~2-5x speedup over WASM backend

## Files

| File | Purpose |
|------|---------|
| `index.html` | Main page with Canvas and controls |
| `hmc-viz.ts` | Main visualization orchestration, sampler runner, debug polling |
| `distributions.ts` | Distribution definitions with true parameters |
| `renderer.ts` | Canvas rendering with zoom support |
| `contour.ts` | Marching squares contour extraction |
| `debug-api.ts` | Vite plugin for REST debug endpoints |
| `console-bridge.ts` | Browser-to-terminal console forwarding |
