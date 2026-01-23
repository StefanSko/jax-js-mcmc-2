# HMC Visualization

Interactive browser-based visualization of Hamiltonian Monte Carlo sampling using JAX-JS.

## Running

From the project root:

```bash
npx vite examples/visualization
```

Then open http://localhost:5173 in your browser.

## Features

### Distributions
- **2D Gaussian** - Standard normal with mild correlation (simple)
- **Banana** - Rosenbrock-like curved posterior (educational)
- **Funnel** - Neal's funnel with varying scales (challenging)

### Controls
- **Step Size** - Leapfrog integration step size (larger = more aggressive proposals)
- **Integration Steps** - Number of leapfrog steps per HMC iteration
- **Animation Speed** - Milliseconds between samples

### Visualization
- **Green dots** - Accepted samples
- **Red dots** - Rejected samples
- **Yellow dots** - Divergent transitions
- **Blue dot** - Current position
- **Gray contours** - Log-density contour lines

## Educational Notes

### Step Size Effects
- Too small: Slow exploration, random walk behavior
- Too large: High rejection rate, divergences
- Optimal: ~65-80% acceptance rate

### Integration Steps
- More steps = larger proposals
- But more computation per sample
- Trade-off between exploration and efficiency

### Divergences
Divergent transitions (yellow) indicate numerical instability, usually from:
- Step size too large for the local curvature
- Funnel distribution has extreme scale variation

## Technical Notes

- JAX-JS has a ~2GB warmup cost for autodiff compilation
- First few samples may be slow as JAX-JS JITs the computation
- Samples are limited to 500 to prevent memory issues
- Contours use pure JavaScript (marching squares) for speed

## Files

- `index.html` - Main page with Canvas and controls
- `hmc-viz.ts` - Main visualization orchestration
- `distributions.ts` - Distribution definitions (log-density functions)
- `renderer.ts` - Canvas rendering utilities
- `contour.ts` - Marching squares contour extraction
