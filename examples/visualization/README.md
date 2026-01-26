# HMC Visualization

Interactive browser-based visualization of Hamiltonian Monte Carlo sampling using JAX-JS.

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
| **Step Size (ε)** | Leapfrog integration step size. Larger = more aggressive proposals |
| **Integration Steps (L)** | Number of leapfrog steps per HMC iteration |
| **Animation Speed** | Milliseconds between samples |
| **Zoom** | Buttons or mouse wheel to zoom in/out on the canvas |

### Visualization Legend

- **Green dots** - Accepted samples
- **Red dots** - Rejected samples
- **Yellow dots** - Divergent transitions
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

## API Endpoints (Development)

For debugging, the dev server exposes REST endpoints:

```bash
# Get current status
curl http://localhost:5173/__api/status

# Control playback
curl -X POST http://localhost:5173/__api/play
curl -X POST http://localhost:5173/__api/pause
curl -X POST http://localhost:5173/__api/step
curl -X POST http://localhost:5173/__api/reset

# List distributions
curl http://localhost:5173/__api/distributions

# Set distribution
curl -X POST http://localhost:5173/__api/distribution/banana

# Set parameters
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
| `hmc-viz.ts` | Main visualization orchestration, HMC runner |
| `distributions.ts` | Distribution definitions with true parameters |
| `renderer.ts` | Canvas rendering with zoom support |
| `contour.ts` | Marching squares contour extraction |
