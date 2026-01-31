/**
 * Energy Conservation Visualization
 *
 * Shows how HMC energy conservation relates to proposal quality,
 * and how large step sizes cause divergence.
 *
 * Uses the velocity-verlet integrator step-by-step to record
 * positions and energies at each leapfrog step.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

import { numpy as np, random, grad, init as jaxInit, defaultDevice } from '@jax-js/jax';
import { distributions, type Distribution } from '../distributions';
import {
  computeDensityGrid,
  computeContourLevels,
  extractContours,
  type ContourLine,
  type Bounds,
} from '../contour';
import { TracePlot } from '../shared/trace-plot';
import { createVelocityVerlet } from '../../../src/integrators/velocity-verlet';
import { createGaussianEuclidean } from '../../../src/metrics/gaussian-euclidean';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrajectoryResult {
  positions: [number, number][];
  energies: number[];
  deltaH: number;
  acceptProb: number;
  classification: 'good' | 'warning' | 'divergent';
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentDistribution: Distribution;
let contours: ContourLine[] = [];
let currentPosition: np.Array;
let baseSeed = Date.now();
let trajectoryCounter = 0;

// Latest trajectory data
let trajectory: TrajectoryResult = {
  positions: [],
  energies: [],
  deltaH: 0,
  acceptProb: 1,
  classification: 'good',
};

// ---------------------------------------------------------------------------
// UI Elements
// ---------------------------------------------------------------------------

const contourCanvas = document.getElementById('contour-canvas') as HTMLCanvasElement;
const energyCanvas = document.getElementById('energy-canvas') as HTMLCanvasElement;
const loadingEl = document.getElementById('loading') as HTMLDivElement;
const distributionSelect = document.getElementById('distribution') as HTMLSelectElement;
const stepSizeSlider = document.getElementById('step-size') as HTMLInputElement;
const stepSizeValue = document.getElementById('step-size-value') as HTMLSpanElement;
const numStepsSlider = document.getElementById('num-steps') as HTMLInputElement;
const numStepsValue = document.getElementById('num-steps-value') as HTMLSpanElement;
const newTrajectoryBtn = document.getElementById('new-trajectory') as HTMLButtonElement;
const resetBtn = document.getElementById('reset') as HTMLButtonElement;
const statDeltaH = document.getElementById('stat-delta-h') as HTMLSpanElement;
const statAcceptProb = document.getElementById('stat-accept-prob') as HTMLSpanElement;
const statClassification = document.getElementById('stat-classification') as HTMLSpanElement;
const statDecision = document.getElementById('stat-decision') as HTMLSpanElement;

// ---------------------------------------------------------------------------
// Energy trace plot
// ---------------------------------------------------------------------------

let energyPlot: TracePlot;

// ---------------------------------------------------------------------------
// Canvas setup helpers
// ---------------------------------------------------------------------------

/** Resize a canvas to match its container with device pixel ratio. */
function resizeCanvas(canvas: HTMLCanvasElement): { width: number; height: number } {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  return { width: rect.width, height: rect.height };
}

// ---------------------------------------------------------------------------
// Coordinate mapping for the contour canvas
// ---------------------------------------------------------------------------

const PADDING = 40;

function worldToCanvas(
  x: number,
  y: number,
  bounds: Bounds,
  canvasW: number,
  canvasH: number,
): [number, number] {
  const plotW = canvasW - 2 * PADDING;
  const plotH = canvasH - 2 * PADDING;
  const cx = PADDING + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * plotW;
  const cy = PADDING + ((bounds.yMax - y) / (bounds.yMax - bounds.yMin)) * plotH;
  return [cx, cy];
}

// ---------------------------------------------------------------------------
// Pure JS log-density for contour grid (no JAX-JS overhead)
// ---------------------------------------------------------------------------

function createJSLogdensity(dist: Distribution): (x: number, y: number) => number {
  const name = dist.name;

  if (name === '2D Gaussian') {
    return (x, y) => -0.5 * (x * x + y * y);
  }

  if (name === 'Banana') {
    const a = 1.0;
    const b = 100.0;
    const scale = 0.05;
    return (x, y) => {
      const term1 = (a - x) ** 2;
      const term2 = b * (y - x * x) ** 2;
      return -scale * (term1 + term2);
    };
  }

  // Fallback: uniform (flat)
  return () => 0;
}

// ---------------------------------------------------------------------------
// Contour computation
// ---------------------------------------------------------------------------

function computeContoursForDist(): void {
  const jsLogdensity = createJSLogdensity(currentDistribution);
  const { grid, xs, ys } = computeDensityGrid(jsLogdensity, currentDistribution.bounds, 60);
  const levels = computeContourLevels(grid, 10);
  contours = extractContours(grid, xs, ys, levels);
}

// ---------------------------------------------------------------------------
// Run one leapfrog trajectory step-by-step
// ---------------------------------------------------------------------------

function runTrajectory(): TrajectoryResult {
  const stepSize = parseFloat(stepSizeSlider.value);
  const numSteps = parseInt(numStepsSlider.value, 10);

  console.log(`[ENERGY-VIZ] Running trajectory: stepSize=${stepSize}, numSteps=${numSteps}`);

  // Create metric and integrator fresh each time to avoid stale state
  const inverseMass = np.array([1.0, 1.0]);
  const metric = createGaussianEuclidean(inverseMass);
  const integrator = createVelocityVerlet(currentDistribution.logdensity, metric.kineticEnergy);

  // Sample fresh momentum
  const key = random.key(baseSeed + trajectoryCounter);
  const [keyMomentum] = random.split(key, 2);

  // sampleMomentum consumes both key and position, so .ref position
  const momentum = metric.sampleMomentum(keyMomentum, currentPosition.ref);

  // Compute logdensity and its gradient at the starting position.
  // Both consume position, so we need .ref.
  const logdensityFn = currentDistribution.logdensity;
  const logdensityGradFn = grad(logdensityFn);

  const initLogdensity = logdensityFn(currentPosition.ref);
  const initLogdensityGrad = logdensityGradFn(currentPosition.ref);

  // Compute initial Hamiltonian: H = -logdensity + kinetic
  // kineticEnergy consumes momentum, so .ref it
  const initialKE = metric.kineticEnergy(momentum.ref);
  const H0 = -(initLogdensity.ref.js() as number) + (initialKE.js() as number);

  // Record starting position and energy
  const positions: [number, number][] = [];
  const energies: number[] = [H0];

  const pos0 = currentPosition.ref.js() as number[];
  positions.push([pos0[0]!, pos0[1]!]);

  // Build initial integrator state
  // The integrator will consume state.momentum, state.position and
  // dispose state.logdensity, state.logdensityGrad on each step.
  // We need to hand it owned arrays. position/momentum are our .ref copies.
  let state = {
    position: currentPosition.ref,
    momentum: momentum,
    logdensity: initLogdensity,
    logdensityGrad: initLogdensityGrad,
  };

  // Run leapfrog steps one at a time
  for (let i = 0; i < numSteps; i++) {
    // integrator consumes state.momentum and state.position,
    // and disposes state.logdensity and state.logdensityGrad.
    // It returns a fresh state that we own.
    state = integrator(state, stepSize);

    // Record position — .ref because state.position will be consumed next iteration
    const pos = state.position.ref.js() as number[];
    positions.push([pos[0]!, pos[1]!]);

    // Record energy at this step
    // kineticEnergy consumes momentum, so .ref it
    const ke = metric.kineticEnergy(state.momentum.ref);
    // state.logdensity will be disposed by the next integrator call, so .ref it
    const H = -(state.logdensity.ref.js() as number) + (ke.js() as number);
    energies.push(H);
  }

  // Compute deltaH and acceptance probability
  const finalH = energies[energies.length - 1]!;
  const deltaH = finalH - H0;
  const acceptProb = Math.min(1, Math.exp(-deltaH));

  // Classify
  const absDeltaH = Math.abs(deltaH);
  let classification: 'good' | 'warning' | 'divergent';
  if (absDeltaH < 0.1) {
    classification = 'good';
  } else if (absDeltaH < 1.0) {
    classification = 'warning';
  } else {
    classification = 'divergent';
  }

  // Dispose the final integrator state — we own all four arrays
  state.position.dispose();
  state.momentum.dispose();
  state.logdensity.dispose();
  state.logdensityGrad.dispose();

  console.log(`[ENERGY-VIZ] Trajectory done: deltaH=${deltaH.toFixed(4)}, acceptProb=${acceptProb.toFixed(4)}, class=${classification}`);

  return { positions, energies, deltaH, acceptProb, classification };
}

// ---------------------------------------------------------------------------
// Rendering: Contour canvas (left side)
// ---------------------------------------------------------------------------

function renderContourCanvas(): void {
  const { width: w, height: h } = resizeCanvas(contourCanvas);
  const ctx = contourCanvas.getContext('2d')!;
  const bounds = currentDistribution.bounds;

  // Background
  ctx.fillStyle = '#16213e';
  ctx.fillRect(0, 0, w, h);

  // Draw contour lines
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (const contour of contours) {
    const { points } = contour;
    for (let i = 0; i < points.length; i += 2) {
      const [x1, y1] = points[i]!;
      const [x2, y2] = points[i + 1]!;
      const [cx1, cy1] = worldToCanvas(x1, y1, bounds, w, h);
      const [cx2, cy2] = worldToCanvas(x2, y2, bounds, w, h);
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.stroke();
    }
  }

  // Draw axes labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px system-ui';
  const xSteps = 5;
  for (let i = 0; i <= xSteps; i++) {
    const x = bounds.xMin + (i / xSteps) * (bounds.xMax - bounds.xMin);
    const [cx, cy] = worldToCanvas(x, bounds.yMin, bounds, w, h);
    ctx.fillText(x.toFixed(1), cx - 12, cy + 16);
  }
  const ySteps = 5;
  for (let i = 0; i <= ySteps; i++) {
    const y = bounds.yMin + (i / ySteps) * (bounds.yMax - bounds.yMin);
    const [cx, cy] = worldToCanvas(bounds.xMin, y, bounds, w, h);
    ctx.fillText(y.toFixed(1), cx - 35, cy + 4);
  }

  // Draw trajectory path with energy-colored segments
  const { positions, energies } = trajectory;
  if (positions.length < 2) return;

  const H0 = energies[0]!;

  for (let i = 0; i < positions.length - 1; i++) {
    const dH = Math.abs(energies[i + 1]! - H0);
    let color: string;
    if (dH >= 1.0) {
      color = '#f87171'; // red — divergent
    } else if (dH >= 0.1) {
      color = '#fbbf24'; // yellow — warning
    } else {
      color = '#4ade80'; // green — good
    }

    const [cx1, cy1] = worldToCanvas(positions[i]![0], positions[i]![1], bounds, w, h);
    const [cx2, cy2] = worldToCanvas(positions[i + 1]![0], positions[i + 1]![1], bounds, w, h);

    // Line segment
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx1, cy1);
    ctx.lineTo(cx2, cy2);
    ctx.stroke();

    // Dot at each leapfrog position (not the first — that gets the start marker)
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx2, cy2, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw start position (blue highlight)
  const [startCx, startCy] = worldToCanvas(positions[0]![0], positions[0]![1], bounds, w, h);

  // Outer glow
  ctx.fillStyle = '#60a5fa40';
  ctx.beginPath();
  ctx.arc(startCx, startCy, 12, 0, Math.PI * 2);
  ctx.fill();

  // Inner solid
  ctx.fillStyle = '#60a5fa';
  ctx.beginPath();
  ctx.arc(startCx, startCy, 6, 0, Math.PI * 2);
  ctx.fill();

  // Draw end position (larger, with border matching its energy color)
  if (positions.length > 1) {
    const last = positions[positions.length - 1]!;
    const [endCx, endCy] = worldToCanvas(last[0], last[1], bounds, w, h);
    const endDH = Math.abs(energies[energies.length - 1]! - H0);
    let endColor: string;
    if (endDH >= 1.0) {
      endColor = '#f87171';
    } else if (endDH >= 0.1) {
      endColor = '#fbbf24';
    } else {
      endColor = '#4ade80';
    }

    // Outer ring
    ctx.strokeStyle = endColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(endCx, endCy, 8, 0, Math.PI * 2);
    ctx.stroke();

    // Inner fill
    ctx.fillStyle = endColor;
    ctx.beginPath();
    ctx.arc(endCx, endCy, 5, 0, Math.PI * 2);
    ctx.fill();

    // Arrow from start to end (dashed)
    ctx.strokeStyle = '#94a3b870';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(startCx, startCy);
    ctx.lineTo(endCx, endCy);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// ---------------------------------------------------------------------------
// Rendering: Energy trace plot (right side)
// ---------------------------------------------------------------------------

function renderEnergyPlot(): void {
  if (!energyPlot) return;

  const H0 = trajectory.energies.length > 0 ? trajectory.energies[0]! : 0;
  energyPlot.setReferenceLine({ value: H0, color: '#4ade80', label: 'H\u2080' });
  energyPlot.setData(trajectory.energies);
  energyPlot.resize();
  energyPlot.render();
}

// ---------------------------------------------------------------------------
// Rendering: Stats panel
// ---------------------------------------------------------------------------

function updateStats(): void {
  const { deltaH, acceptProb, classification } = trajectory;

  statDeltaH.textContent = deltaH.toFixed(4);
  statAcceptProb.textContent = acceptProb.toFixed(4);

  // Classification badge
  const badgeClass =
    classification === 'good'
      ? 'classification-good'
      : classification === 'warning'
        ? 'classification-warning'
        : 'classification-divergent';
  const label =
    classification === 'good'
      ? 'Good'
      : classification === 'warning'
        ? 'Warning'
        : 'Divergent';
  statClassification.innerHTML = `<span class="classification-badge ${badgeClass}">${label}</span>`;

  // Accept/reject decision (probabilistic based on acceptProb)
  const accepted = acceptProb >= 1.0 || Math.random() < acceptProb;
  if (accepted) {
    statDecision.textContent = 'Accept';
    statDecision.style.color = '#4ade80';
  } else {
    statDecision.textContent = 'Reject';
    statDecision.style.color = '#f87171';
  }
}

// ---------------------------------------------------------------------------
// Full render cycle
// ---------------------------------------------------------------------------

function render(): void {
  renderContourCanvas();
  renderEnergyPlot();
  updateStats();
}

// ---------------------------------------------------------------------------
// Distribution change: rebuild everything
// ---------------------------------------------------------------------------

function switchDistribution(key: string): void {
  const factory = distributions[key as keyof typeof distributions];
  if (!factory) {
    console.error(`[ENERGY-VIZ] Unknown distribution: ${key}`);
    return;
  }

  currentDistribution = factory();
  console.log(`[ENERGY-VIZ] Switched to distribution: ${currentDistribution.name}`);

  // Dispose old position, create new one
  if (currentPosition) {
    currentPosition.dispose();
  }
  const [ix, iy] = currentDistribution.initialPosition;
  currentPosition = np.array([ix, iy]);

  // Recompute contours
  computeContoursForDist();

  // Reset trajectory counter
  trajectoryCounter = 0;

  // Run trajectory
  trajectory = runTrajectory();
  trajectoryCounter++;

  render();
}

// ---------------------------------------------------------------------------
// New trajectory: increment seed, re-run
// ---------------------------------------------------------------------------

function newTrajectory(): void {
  trajectory = runTrajectory();
  trajectoryCounter++;
  render();
}

// ---------------------------------------------------------------------------
// Reset: restore defaults
// ---------------------------------------------------------------------------

function resetAll(): void {
  stepSizeSlider.value = '0.05';
  stepSizeValue.textContent = '0.05';
  numStepsSlider.value = '20';
  numStepsValue.textContent = '20';

  baseSeed = Date.now();
  trajectoryCounter = 0;

  // Reset position
  if (currentPosition) {
    currentPosition.dispose();
  }
  const [ix, iy] = currentDistribution.initialPosition;
  currentPosition = np.array([ix, iy]);

  // Run trajectory
  trajectory = runTrajectory();
  trajectoryCounter++;

  render();
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

function setupEventListeners(): void {
  // Step size slider
  stepSizeSlider.addEventListener('input', () => {
    stepSizeValue.textContent = stepSizeSlider.value;
  });
  stepSizeSlider.addEventListener('change', () => {
    trajectory = runTrajectory();
    trajectoryCounter++;
    render();
  });

  // Num steps slider
  numStepsSlider.addEventListener('input', () => {
    numStepsValue.textContent = numStepsSlider.value;
  });
  numStepsSlider.addEventListener('change', () => {
    trajectory = runTrajectory();
    trajectoryCounter++;
    render();
  });

  // Distribution change
  distributionSelect.addEventListener('change', () => {
    switchDistribution(distributionSelect.value);
  });

  // Buttons
  newTrajectoryBtn.addEventListener('click', () => {
    newTrajectory();
  });
  resetBtn.addEventListener('click', () => {
    resetAll();
  });

  // Resize handler
  window.addEventListener('resize', () => {
    render();
  });
}

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

interface VizDebugAPI {
  getState: () => {
    stepSize: number;
    numSteps: number;
    distribution: string;
    trajectory: { positions: [number, number][]; energies: number[] };
    deltaH: number;
    acceptProb: number;
    classification: string;
  };
  step: () => ReturnType<VizDebugAPI['getState']>;
  reset: () => ReturnType<VizDebugAPI['getState']>;
  setConfig: (config: {
    stepSize?: number;
    numSteps?: number;
    distribution?: string;
  }) => ReturnType<VizDebugAPI['getState']>;
}

function getDebugState() {
  return {
    stepSize: parseFloat(stepSizeSlider.value),
    numSteps: parseInt(numStepsSlider.value, 10),
    distribution: currentDistribution.name,
    trajectory: {
      positions: trajectory.positions,
      energies: trajectory.energies,
    },
    deltaH: trajectory.deltaH,
    acceptProb: trajectory.acceptProb,
    classification: trajectory.classification,
  };
}

const vizDebugAPI: VizDebugAPI = {
  getState: getDebugState,

  step: () => {
    newTrajectory();
    return getDebugState();
  },

  reset: () => {
    resetAll();
    return getDebugState();
  },

  setConfig: (config) => {
    if (config.stepSize !== undefined) {
      stepSizeSlider.value = String(config.stepSize);
      stepSizeValue.textContent = String(config.stepSize);
    }
    if (config.numSteps !== undefined) {
      numStepsSlider.value = String(config.numSteps);
      numStepsValue.textContent = String(config.numSteps);
    }
    if (config.distribution !== undefined) {
      distributionSelect.value = config.distribution;
      switchDistribution(config.distribution);
      return getDebugState();
    }

    // Re-run with updated config
    trajectory = runTrajectory();
    trajectoryCounter++;
    render();
    return getDebugState();
  },
};

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = vizDebugAPI;
console.log('[ENERGY-VIZ] Debug API exposed to window.__vizDebug');

// ---------------------------------------------------------------------------
// Debug command polling loop (dev mode only)
// ---------------------------------------------------------------------------

interface DebugCommand {
  id: string;
  type: string;
  payload?: unknown;
}

async function debugPollLoop(): Promise<void> {
  if (import.meta.env.PROD) return;

  while (true) {
    try {
      const res = await fetch('/__debug/poll');
      const cmd = (await res.json()) as DebugCommand | null;

      if (cmd?.type) {
        const api = (window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug;
        let result: unknown;

        switch (cmd.type) {
          case 'getState':
            result = api.getState();
            break;
          case 'step':
            result = api.step();
            break;
          case 'reset':
            result = api.reset();
            break;
          case 'setConfig':
            result = api.setConfig(cmd.payload as {
              stepSize?: number;
              numSteps?: number;
              distribution?: string;
            });
            break;
          default:
            result = { error: `Unknown command: ${cmd.type}` };
        }

        await fetch('/__debug/result', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: cmd.id, result }),
        });
      }
    } catch {
      // Ignore errors (server may not be available)
    }

    await new Promise((r) => setTimeout(r, 100));
  }
}

// Start polling loop
debugPollLoop();

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

async function init(): Promise<void> {
  console.log('[ENERGY-VIZ] 1. init() starting');
  loadingEl.querySelector('.loading-text')!.textContent = 'Initializing JAX-JS...';

  // Let UI render
  await new Promise((r) => setTimeout(r, 100));

  try {
    // Initialize JAX-JS backends
    console.log('[ENERGY-VIZ] 1a. Initializing JAX-JS backends...');
    const availableDevices = await jaxInit();
    console.log('[ENERGY-VIZ] 1b. Available devices:', availableDevices);

    if (availableDevices.includes('webgpu')) {
      defaultDevice('webgpu');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WebGPU backend...';
      console.log('[ENERGY-VIZ] 1c. Using WebGPU backend');
    } else {
      console.warn('[ENERGY-VIZ] 1c. WebGPU not available, using wasm backend');
      loadingEl.querySelector('.loading-text')!.textContent =
        'Using WASM backend (WebGPU unavailable)...';
    }

    await new Promise((r) => setTimeout(r, 50));

    // Set up distribution
    currentDistribution = distributions.gaussian!();
    console.log('[ENERGY-VIZ] 2. Distribution created:', currentDistribution.name);

    // Set initial position
    const [ix, iy] = currentDistribution.initialPosition;
    currentPosition = np.array([ix, iy]);
    console.log('[ENERGY-VIZ] 3. Initial position:', [ix, iy]);

    // Compute contours
    loadingEl.querySelector('.loading-text')!.textContent = 'Computing contours...';
    await new Promise((r) => setTimeout(r, 50));
    computeContoursForDist();
    console.log('[ENERGY-VIZ] 4. Contours computed');

    // Create energy trace plot
    energyPlot = new TracePlot(energyCanvas, {
      yLabel: 'H(q,p)',
      color: '#60a5fa',
      referenceLine: { value: 0, color: '#4ade80', label: 'H\u2080' },
    });
    console.log('[ENERGY-VIZ] 5. Energy plot created');

    // Run first trajectory
    loadingEl.querySelector('.loading-text')!.textContent = 'Running trajectory...';
    await new Promise((r) => setTimeout(r, 50));
    trajectory = runTrajectory();
    trajectoryCounter++;
    console.log('[ENERGY-VIZ] 6. First trajectory computed');

    // Render
    render();
    console.log('[ENERGY-VIZ] 7. Initial render done');

    // Set up event listeners
    setupEventListeners();

    // Hide loading overlay
    loadingEl.style.display = 'none';
    console.log('[ENERGY-VIZ] 8. Complete!');
  } catch (error) {
    loadingEl.querySelector('.loading-text')!.textContent = `Error: ${
      error instanceof Error ? error.message : 'Unknown error'
    }`;
    console.error('[ENERGY-VIZ] Initialization error:', error);
  }
}

// Start
init();
