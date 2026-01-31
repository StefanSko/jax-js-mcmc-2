/**
 * RWM Explorer Visualization - Main Module
 *
 * Interactive browser visualization of Random Walk Metropolis sampling.
 * Focused on demonstrating step size sensitivity.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

import { numpy as np, random, init as jaxInit, defaultDevice } from '@jax-js/jax';
import { RWM, type RWMInfo, type RWMState } from '../../../src';
import { distributions, type Distribution } from '../distributions';
import { computeDensityGrid, computeContourLevels, extractContours, type ContourLine } from '../contour';
import { CanvasRenderer, type Sample } from '../renderer';
import { TracePlot } from '../shared/trace-plot';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentDistribution: Distribution;
type Sampler = {
  init: (position: np.Array) => RWMState;
  step: (key: np.Array, state: RWMState) => [RWMState, RWMInfo];
};

let sampler: Sampler;
let samplerState: RWMState;
let samples: Sample[] = [];
let contours: ContourLine[] = [];
let renderer: CanvasRenderer;
let tracePlot: TracePlot;
let isPlaying = false;
let animationTimer: number | null = null;
let baseSeed = Date.now();
let stepCounter = 0;
let acceptedCount = 0;
let prevX: number | undefined;
let prevY: number | undefined;

// ---------------------------------------------------------------------------
// UI Elements
// ---------------------------------------------------------------------------

const contourCanvas = document.getElementById('contour-canvas') as HTMLCanvasElement;
const traceCanvas = document.getElementById('trace-canvas') as HTMLCanvasElement;
const loadingEl = document.getElementById('loading') as HTMLDivElement;
const distributionSelect = document.getElementById('distribution') as HTMLSelectElement;
const stepSizeSlider = document.getElementById('step-size') as HTMLInputElement;
const stepSizeValue = document.getElementById('step-size-value') as HTMLSpanElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const speedValue = document.getElementById('speed-value') as HTMLSpanElement;
const playPauseBtn = document.getElementById('play-pause') as HTMLButtonElement;
const stepBtn = document.getElementById('step-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

// Stats elements
const statSamples = document.getElementById('stat-samples') as HTMLSpanElement;
const statAcceptance = document.getElementById('stat-acceptance') as HTMLSpanElement;
const statMeanX = document.getElementById('stat-mean-x') as HTMLSpanElement;
const statMeanY = document.getElementById('stat-mean-y') as HTMLSpanElement;
const infoAcceptProb = document.getElementById('info-accept-prob') as HTMLSpanElement;
const infoStatus = document.getElementById('info-status') as HTMLSpanElement;

// ---------------------------------------------------------------------------
// Pure JS log-density evaluators for contour computation
// ---------------------------------------------------------------------------

function createJSLogdensity(dist: Distribution): (x: number, y: number) => number {
  const name = dist.name;

  if (name === '2D Gaussian') {
    return (x: number, y: number) => -0.5 * (x * x + y * y);
  }

  if (name === 'Banana') {
    const a = 1.0;
    const b = 100.0;
    const scale = 0.05;
    return (x: number, y: number) => {
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

function computeContours(): void {
  const jsLogdensity = createJSLogdensity(currentDistribution);
  const { grid, xs, ys } = computeDensityGrid(jsLogdensity, currentDistribution.bounds, 60);
  const levels = computeContourLevels(grid, 10);
  contours = extractContours(grid, xs, ys, levels);
}

// ---------------------------------------------------------------------------
// Dispose RWMInfo arrays to prevent memory leaks
// ---------------------------------------------------------------------------

function disposeInfo(info: RWMInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.proposedPosition.dispose();
}

// ---------------------------------------------------------------------------
// Build sampler with current parameters
// ---------------------------------------------------------------------------

function buildSampler(): void {
  console.log('[RWM-VIZ] buildSampler() entered');
  const stepSize = parseFloat(stepSizeSlider.value);
  console.log('[RWM-VIZ] Parsed params:', { stepSize });

  sampler = RWM(currentDistribution.logdensity).stepSize(stepSize).build();
  console.log('[RWM-VIZ] buildSampler() complete');
}

// ---------------------------------------------------------------------------
// Initialize sampler state at distribution's initial position
// ---------------------------------------------------------------------------

function initializeState(): void {
  const [x, y] = currentDistribution.initialPosition;
  samplerState = sampler.init(np.array([x, y]));
  prevX = undefined;
  prevY = undefined;
}

// ---------------------------------------------------------------------------
// Perform one sampler step
// ---------------------------------------------------------------------------

// Last step info for debug API
let lastStepResult: {
  accepted: boolean;
  acceptanceProb: number;
  position: [number, number];
} | null = null;

function performStep(): void {
  // Save old position BEFORE step (step consumes the state)
  const oldPos = samplerState.position.ref.js() as number[];
  prevX = oldPos[0];
  prevY = oldPos[1];

  const key = random.key(baseSeed + stepCounter);
  const [newState, info] = sampler.step(key, samplerState);

  // Read info values before disposal
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;

  // Debug logging
  console.log(`[RWM-VIZ] Step ${stepCounter}: pos=(${prevX?.toFixed(2)}, ${prevY?.toFixed(2)}) -> acceptProb=${acceptanceProb.toFixed(4)}, accepted=${isAccepted}`);

  // Capture for debug API
  const newPosArray = newState.position.ref.js() as number[];
  lastStepResult = {
    accepted: isAccepted,
    acceptanceProb,
    position: [newPosArray[0]!, newPosArray[1]!],
  };

  // Dispose info (old state was consumed by step)
  disposeInfo(info);

  // Update state
  samplerState = newState;
  stepCounter++;

  // Get new position
  const posArray = newState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;

  // Track sample
  samples.push({ x, y, accepted: isAccepted });
  if (isAccepted) acceptedCount++;

  // Add x-coordinate to trace plot
  tracePlot.addPoint(x);

  // Limit samples to prevent memory issues
  if (samples.length > 500) {
    samples = samples.slice(-500);
  }

  // Update UI
  updateStats();
  updateCurrentInfo({ acceptanceProb, isAccepted });
  render();
}

// ---------------------------------------------------------------------------
// Update statistics display
// ---------------------------------------------------------------------------

function updateStats(): void {
  statSamples.textContent = String(stepCounter);

  if (stepCounter > 0) {
    const rate = (acceptedCount / stepCounter) * 100;
    statAcceptance.textContent = `${rate.toFixed(1)}%`;

    const meanX = samples.reduce((sum, s) => sum + s.x, 0) / samples.length;
    const meanY = samples.reduce((sum, s) => sum + s.y, 0) / samples.length;
    statMeanX.textContent = meanX.toFixed(3);
    statMeanY.textContent = meanY.toFixed(3);
  } else {
    statAcceptance.textContent = '-';
    statMeanX.textContent = '-';
    statMeanY.textContent = '-';
  }
}

// ---------------------------------------------------------------------------
// Update current step info display
// ---------------------------------------------------------------------------

function updateCurrentInfo(info: {
  acceptanceProb: number;
  isAccepted: boolean;
} | null): void {
  if (info === null) {
    infoAcceptProb.textContent = '-';
    infoStatus.textContent = '-';
    infoStatus.style.color = '';
    return;
  }

  infoAcceptProb.textContent = info.acceptanceProb.toFixed(3);

  if (info.isAccepted) {
    infoStatus.textContent = 'Accepted';
    infoStatus.style.color = '#4ade80';
  } else {
    infoStatus.textContent = 'Rejected';
    infoStatus.style.color = '#f87171';
  }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render(): void {
  const posArray = samplerState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;
  renderer.render(contours, samples, x, y, prevX, prevY);
  tracePlot.render();
}

// ---------------------------------------------------------------------------
// Reset everything for current distribution
// ---------------------------------------------------------------------------

function reset(): void {
  // Stop animation
  if (animationTimer !== null) {
    clearTimeout(animationTimer);
    animationTimer = null;
  }
  isPlaying = false;
  playPauseBtn.textContent = 'Play';

  // Reset stats
  samples = [];
  baseSeed = Date.now();
  stepCounter = 0;
  acceptedCount = 0;

  // Clear trace plot
  tracePlot.clear();

  // Rebuild sampler and state
  buildSampler();
  initializeState();
  computeContours();

  // Update renderer bounds
  renderer.setBounds(currentDistribution.bounds);

  // Update UI
  updateStats();
  updateCurrentInfo(null);
  render();
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------

function tick(): void {
  if (!isPlaying) return;

  try {
    performStep();
  } catch (error) {
    console.error('[RWM-VIZ] tick() error:', error);
    isPlaying = false;
    playPauseBtn.textContent = 'Play';
    return;
  }

  const speed = parseInt(speedSlider.value, 10);
  animationTimer = window.setTimeout(tick, speed);
}

function togglePlay(): void {
  isPlaying = !isPlaying;
  playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';

  if (isPlaying) {
    tick();
  } else if (animationTimer !== null) {
    clearTimeout(animationTimer);
    animationTimer = null;
  }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

function setupEventListeners(): void {
  // Distribution change
  distributionSelect.addEventListener('change', () => {
    const key = distributionSelect.value as keyof typeof distributions;
    currentDistribution = distributions[key]!();
    reset();
  });

  // Step size slider — live display update
  stepSizeSlider.addEventListener('input', () => {
    stepSizeValue.textContent = parseFloat(stepSizeSlider.value).toFixed(2);
  });

  // Step size slider — rebuild sampler on commit
  stepSizeSlider.addEventListener('change', () => {
    buildSampler();
    initializeState();
  });

  // Speed slider
  speedSlider.addEventListener('input', () => {
    speedValue.textContent = `${speedSlider.value}ms`;
  });

  // Buttons
  playPauseBtn.addEventListener('click', togglePlay);
  stepBtn.addEventListener('click', performStep);
  resetBtn.addEventListener('click', reset);

  // Resize handler — update both canvases
  window.addEventListener('resize', () => {
    renderer.resize();
    tracePlot.resize();
    render();
  });
}

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

interface VizDebugAPI {
  getState: () => {
    distribution: string;
    position: [number, number];
    stepCount: number;
    acceptanceRate: number;
    config: { stepSize: number };
  };
  step: () => {
    accepted: boolean;
    acceptanceProb: number;
    position: [number, number];
  };
  reset: () => { ok: true; position: [number, number] };
  setConfig: (config: { stepSize?: number; distribution?: string }) => {
    stepSize: number;
    distribution: string;
  };
}

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = {
  getState: () => {
    const posArray = samplerState.position.ref.js() as number[];
    return {
      distribution: currentDistribution.name,
      position: [posArray[0]!, posArray[1]!],
      stepCount: stepCounter,
      acceptanceRate: stepCounter > 0 ? acceptedCount / stepCounter : 0,
      config: {
        stepSize: parseFloat(stepSizeSlider.value),
      },
    };
  },
  step: () => {
    if (isPlaying) {
      togglePlay(); // Pause first
    }
    performStep();
    return lastStepResult!;
  },
  reset: () => {
    reset();
    return { ok: true as const, position: currentDistribution.initialPosition as [number, number] };
  },
  setConfig: (config: { stepSize?: number; distribution?: string }) => {
    if (config.distribution !== undefined) {
      const key = Object.keys(distributions).find(
        (k) => distributions[k as keyof typeof distributions]?.().name === config.distribution ||
               k === config.distribution
      );
      if (key) {
        currentDistribution = distributions[key as keyof typeof distributions]!();
        distributionSelect.value = key;
      }
    }
    if (config.stepSize !== undefined) {
      stepSizeSlider.value = String(config.stepSize);
      stepSizeValue.textContent = config.stepSize.toFixed(2);
    }
    buildSampler();
    initializeState();
    computeContours();
    renderer.setBounds(currentDistribution.bounds);
    tracePlot.clear();
    render();
    console.log('[RWM-VIZ] Config updated:', config);
    return {
      stepSize: parseFloat(stepSizeSlider.value),
      distribution: currentDistribution.name,
    };
  },
};

console.log('[RWM-VIZ] Debug API exposed to window.__vizDebug');

// ---------------------------------------------------------------------------
// Debug command polling loop (dev mode only)
// ---------------------------------------------------------------------------

type DebugConfig = { stepSize?: number; distribution?: string };

async function debugPollLoop(): Promise<void> {
  if (import.meta.env.PROD) return;

  while (true) {
    try {
      const res = await fetch('/__debug/poll');
      const cmd = await res.json() as { id: string; type: string; payload?: unknown } | null;

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
            result = api.setConfig(cmd.payload as DebugConfig);
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
// Initialization
// ---------------------------------------------------------------------------

async function init(): Promise<void> {
  console.log('[RWM-VIZ] 1. init() starting');
  loadingEl.querySelector('.loading-text')!.textContent = 'Initializing JAX-JS...';

  // Small delay to let UI render
  await new Promise((r) => setTimeout(r, 100));

  try {
    // Initialize JAX-JS with WebGPU (fall back to wasm if unavailable)
    console.log('[RWM-VIZ] 1a. Initializing JAX-JS backends...');
    const availableDevices = await jaxInit();
    console.log('[RWM-VIZ] 1b. Available devices:', availableDevices);

    if (availableDevices.includes('webgpu')) {
      defaultDevice('webgpu');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WebGPU backend...';
      console.log('[RWM-VIZ] 1c. Using WebGPU backend');
    } else {
      console.warn('[RWM-VIZ] 1c. WebGPU not available, using wasm backend');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WASM backend (WebGPU unavailable)...';
    }

    await new Promise((r) => setTimeout(r, 50));

    // Initialize distribution
    currentDistribution = distributions.gaussian!();
    console.log('[RWM-VIZ] 2. Distribution created:', currentDistribution.name);

    // Create renderer
    renderer = new CanvasRenderer(contourCanvas, currentDistribution.bounds);
    console.log('[RWM-VIZ] 3. Renderer created');

    // Create trace plot
    tracePlot = new TracePlot(traceCanvas, { yLabel: 'x\u2081', color: '#60a5fa' });
    console.log('[RWM-VIZ] 3a. TracePlot created');

    // Build sampler and state (this triggers JAX-JS warmup)
    loadingEl.querySelector('.loading-text')!.textContent = 'Building sampler...';
    await new Promise((r) => setTimeout(r, 50));

    console.log('[RWM-VIZ] 4. Building sampler...');
    buildSampler();
    console.log('[RWM-VIZ] 5. Sampler built');
    initializeState();
    console.log('[RWM-VIZ] 6. State initialized');

    // Compute contours
    loadingEl.querySelector('.loading-text')!.textContent = 'Computing contours...';
    await new Promise((r) => setTimeout(r, 50));
    console.log('[RWM-VIZ] 7. Computing contours...');
    computeContours();
    console.log('[RWM-VIZ] 8. Contours computed');

    // Initial render
    console.log('[RWM-VIZ] 9. Rendering...');
    render();

    // Setup UI
    setupEventListeners();

    // Hide loading
    loadingEl.style.display = 'none';
    console.log('[RWM-VIZ] 10. Complete!');
  } catch (error) {
    loadingEl.querySelector('.loading-text')!.textContent =
      `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    console.error('[RWM-VIZ] Initialization error:', error);
  }
}

// Start
init();
