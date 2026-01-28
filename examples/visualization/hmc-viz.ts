/**
 * HMC Visualization - Main Module
 *
 * Interactive browser visualization of Hamiltonian Monte Carlo sampling.
 */

// Console bridge must be first - sends browser logs to terminal
import './console-bridge';

import { numpy as np, random, init as jaxInit, defaultDevice } from '@jax-js/jax';
import { HMC, RWM, type HMCInfo, type HMCState, type RWMInfo, type RWMState } from '../../src';
import { distributions, type Distribution } from './distributions';
import { computeDensityGrid, computeContourLevels, extractContours, type ContourLine } from './contour';
import { CanvasRenderer, type Sample } from './renderer';

// State
let currentDistribution: Distribution;
type Algorithm = 'hmc' | 'rwm';
type SamplerState = HMCState | RWMState;
type SamplerInfo = HMCInfo | RWMInfo;
type Sampler = {
  init: (position: np.Array) => SamplerState;
  step: (key: np.Array, state: SamplerState) => [SamplerState, SamplerInfo];
};

let currentAlgorithm: Algorithm = 'hmc';
let sampler: Sampler;
let samplerState: SamplerState;
let samples: Sample[] = [];
let contours: ContourLine[] = [];
let renderer: CanvasRenderer;
let isPlaying = false;
let animationTimer: number | null = null;
let baseSeed = Date.now();  // Random base seed, changes on each reset
let stepCounter = 0;
let acceptedCount = 0;
let divergentCount = 0;
let prevX: number | undefined;
let prevY: number | undefined;

// UI Elements
const canvas = document.getElementById('hmc-canvas') as HTMLCanvasElement;
const loadingEl = document.getElementById('loading') as HTMLDivElement;
const algorithmSelect = document.getElementById('algorithm') as HTMLSelectElement;
const distributionSelect = document.getElementById('distribution') as HTMLSelectElement;
const stepSizeSlider = document.getElementById('step-size') as HTMLInputElement;
const stepSizeValue = document.getElementById('step-size-value') as HTMLSpanElement;
const numStepsGroup = document.getElementById('num-steps-group') as HTMLDivElement;
const numStepsSlider = document.getElementById('num-steps') as HTMLInputElement;
const numStepsValue = document.getElementById('num-steps-value') as HTMLSpanElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const speedValue = document.getElementById('speed-value') as HTMLSpanElement;
const playPauseBtn = document.getElementById('play-pause') as HTMLButtonElement;
const stepBtn = document.getElementById('step') as HTMLButtonElement;
const resetBtn = document.getElementById('reset') as HTMLButtonElement;

// True params elements
const showTrueParamsCheckbox = document.getElementById('show-true-params') as HTMLInputElement;
const trueParamsInfo = document.getElementById('true-params-info') as HTMLDivElement;

// Zoom elements
const zoomInBtn = document.getElementById('zoom-in') as HTMLButtonElement;
const zoomOutBtn = document.getElementById('zoom-out') as HTMLButtonElement;
const zoomResetBtn = document.getElementById('zoom-reset') as HTMLButtonElement;
const zoomLevelDisplay = document.getElementById('zoom-level') as HTMLSpanElement;

// Stats elements
const statSamples = document.getElementById('stat-samples') as HTMLSpanElement;
const statAcceptance = document.getElementById('stat-acceptance') as HTMLSpanElement;
const statDivergent = document.getElementById('stat-divergent') as HTMLSpanElement;
const statMeanX = document.getElementById('stat-mean-x') as HTMLSpanElement;
const statMeanY = document.getElementById('stat-mean-y') as HTMLSpanElement;
const infoAcceptProb = document.getElementById('info-accept-prob') as HTMLSpanElement;
const infoEnergy = document.getElementById('info-energy') as HTMLSpanElement;
const infoStatus = document.getElementById('info-status') as HTMLSpanElement;

/**
 * Dispose sampler info arrays to prevent memory leaks.
 */
function disposeInfo(info: SamplerInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();

  if ('momentum' in info) {
    info.momentum.dispose();
    info.isDivergent.dispose();
    info.energy.dispose();
  } else {
    info.proposedPosition.dispose();
  }
}

/**
 * Create a pure JS log-density evaluator for contour computation.
 * This avoids JAX-JS overhead for grid evaluation.
 */
function createJSLogdensity(dist: Distribution): (x: number, y: number) => number {
  const name = dist.name;

  if (name === '2D Gaussian') {
    // Isotropic Gaussian: log p(x) = -0.5 * ||x||^2
    return (x: number, y: number) => -0.5 * (x * x + y * y);
  }

  if (name === 'Correlated Gaussian') {
    // Correlated Gaussian with rho = 0.9
    const rho = 0.9;
    const factor = 1 / (1 - rho * rho);
    return (x: number, y: number) => {
      const sumSq = x * x + y * y;
      const xy2 = 2 * x * y;
      const quadForm = factor * sumSq - rho * factor * xy2;
      return -0.5 * quadForm;
    };
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

  if (name === 'Bimodal') {
    const mode1 = [-2, 0];
    const mode2 = [2, 0];
    const sigma = 0.7;
    const sigmaSq2 = 2 * sigma * sigma;
    return (x: number, y: number) => {
      const d1Sq = (x - mode1[0]) ** 2 + (y - mode1[1]) ** 2;
      const d2Sq = (x - mode2[0]) ** 2 + (y - mode2[1]) ** 2;
      const logp1 = -d1Sq / sigmaSq2;
      const logp2 = -d2Sq / sigmaSq2;
      // log-sum-exp
      const maxLog = Math.max(logp1, logp2);
      return maxLog + Math.log(Math.exp(logp1 - maxLog) + Math.exp(logp2 - maxLog));
    };
  }

  if (name === 'Donut') {
    const radius = 2.5;
    const width = 0.4;
    const widthSq2 = 2 * width * width;
    return (x: number, y: number) => {
      const r = Math.sqrt(x * x + y * y);
      const deviation = r - radius;
      return -(deviation * deviation) / widthSq2;
    };
  }

  if (name === 'Funnel') {
    return (v: number, x: number) => {
      const logPv = -(v * v) / 18;
      const logPxGivenV = -v / 2 - (x * x) / (2 * Math.exp(v));
      return logPv + logPxGivenV;
    };
  }

  if (name === 'Squiggle') {
    const amplitude = 1.5;
    const frequency = 1.0;
    const width = 0.3;
    const widthSq2 = 2 * width * width;
    return (x: number, y: number) => {
      const ridge = amplitude * Math.sin(frequency * x);
      const deviation = y - ridge;
      return -(deviation * deviation) / widthSq2;
    };
  }

  // Fallback: uniform (flat)
  return () => 0;
}

/**
 * Compute contours for the current distribution.
 */
function computeContours(): void {
  const jsLogdensity = createJSLogdensity(currentDistribution);
  const { grid, xs, ys } = computeDensityGrid(jsLogdensity, currentDistribution.bounds, 60);
  const levels = computeContourLevels(grid, 10);
  contours = extractContours(grid, xs, ys, levels);
}

type SamplerConfig = { stepSize: number; numIntegrationSteps: number };

function createSampler(
  algorithm: Algorithm,
  logdensityFn: (q: np.Array) => np.Array,
  config: SamplerConfig
): Sampler {
  if (algorithm === 'rwm') {
    return RWM(logdensityFn).stepSize(config.stepSize).build();
  }

  return HMC(logdensityFn)
    .stepSize(config.stepSize)
    .numIntegrationSteps(config.numIntegrationSteps)
    .inverseMassMatrix(np.array([1.0, 1.0]))
    .build();
}

/**
 * Build sampler with current parameters.
 */
function buildSampler(): void {
  console.log('[HMC-VIZ] 4a. buildSampler() entered');
  const stepSize = parseFloat(stepSizeSlider.value);
  const numSteps = parseInt(numStepsSlider.value, 10);
  console.log('[HMC-VIZ] 4b. Parsed params:', { stepSize, numSteps, algorithm: currentAlgorithm });

  sampler = createSampler(currentAlgorithm, currentDistribution.logdensity, {
    stepSize,
    numIntegrationSteps: numSteps,
  });
  console.log('[HMC-VIZ] 4c. buildSampler() complete');
}

function syncAlgorithmUI(): void {
  const isHmc = currentAlgorithm === 'hmc';
  numStepsGroup.classList.toggle('hidden', !isHmc);
  numStepsSlider.disabled = !isHmc;
}

/**
 * Initialize sampler state at distribution's initial position.
 */
function initializeState(): void {
  const [x, y] = currentDistribution.initialPosition;
  samplerState = sampler.init(np.array([x, y]));
  prevX = undefined;
  prevY = undefined;
}

/**
 * Reset everything for current distribution.
 */
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
  baseSeed = Date.now();  // New random seed for each reset
  stepCounter = 0;
  acceptedCount = 0;
  divergentCount = 0;

  syncAlgorithmUI();

  // Rebuild sampler and state
  buildSampler();
  initializeState();
  computeContours();

  // Update renderer bounds
  renderer.setBounds(currentDistribution.bounds);

  // Update UI
  updateStats();
  updateCurrentInfo(null);
  trueParamsInfo.textContent = currentDistribution.trueParams.description;
  render();
}

/**
 * Perform one sampler step.
 */
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
  const isDivergent = 'isDivergent' in info
    ? (info.isDivergent.ref.js() as boolean)
    : false;
  const energy = 'energy' in info ? (info.energy.ref.js() as number) : null;

  // Debug logging
  const energyLabel = energy === null ? 'N/A' : energy.toFixed(2);
  console.log(`[HMC-VIZ] Step ${stepCounter}: pos=(${prevX?.toFixed(2)}, ${prevY?.toFixed(2)}) -> acceptProb=${acceptanceProb.toFixed(4)}, energy=${energyLabel}, accepted=${isAccepted}`);

  // Capture for debug API (position will be updated after state update)
  const newPosArray = newState.position.ref.js() as number[];
  lastStepResult = {
    accepted: isAccepted,
    acceptanceProb,
    position: [newPosArray[0]!, newPosArray[1]!],
    ...(energy !== null && { energy }),
    ...(currentAlgorithm === 'hmc' && { isDivergent }),
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
  samples.push({ x, y, accepted: isAccepted, divergent: isDivergent });
  if (isAccepted) acceptedCount++;
  if (isDivergent) divergentCount++;

  // Limit samples to prevent memory issues
  if (samples.length > 500) {
    samples = samples.slice(-500);
  }

  // Update UI
  updateStats();
  updateCurrentInfo({ acceptanceProb, isAccepted, isDivergent, energy });
  render();
}

function formatDivergentCount(count: number): string {
  return currentAlgorithm === 'hmc' ? String(count) : 'N/A';
}

/**
 * Update statistics display.
 */
function updateStats(): void {
  statSamples.textContent = String(samples.length);

  if (samples.length > 0) {
    const rate = (acceptedCount / samples.length) * 100;
    statAcceptance.textContent = `${rate.toFixed(1)}%`;
    statDivergent.textContent = formatDivergentCount(divergentCount);

    const meanX = samples.reduce((sum, s) => sum + s.x, 0) / samples.length;
    const meanY = samples.reduce((sum, s) => sum + s.y, 0) / samples.length;
    statMeanX.textContent = meanX.toFixed(3);
    statMeanY.textContent = meanY.toFixed(3);
  } else {
    statAcceptance.textContent = '-';
    statDivergent.textContent = formatDivergentCount(0);
    statMeanX.textContent = '-';
    statMeanY.textContent = '-';
  }
}

/**
 * Update current step info display.
 */
function updateCurrentInfo(info: {
  acceptanceProb: number;
  isAccepted: boolean;
  isDivergent: boolean;
  energy: number | null;
} | null): void {
  if (info === null) {
    infoAcceptProb.textContent = '-';
    infoEnergy.textContent = currentAlgorithm === 'rwm' ? 'N/A' : '-';
    infoStatus.textContent = '-';
    return;
  }

  infoAcceptProb.textContent = info.acceptanceProb.toFixed(3);
  infoEnergy.textContent = info.energy === null ? 'N/A' : info.energy.toFixed(2);

  if (info.isDivergent) {
    infoStatus.textContent = 'Divergent';
    infoStatus.style.color = '#facc15';
  } else if (info.isAccepted) {
    infoStatus.textContent = 'Accepted';
    infoStatus.style.color = '#4ade80';
  } else {
    infoStatus.textContent = 'Rejected';
    infoStatus.style.color = '#f87171';
  }
}

/**
 * Update zoom level display.
 */
function updateZoomDisplay(): void {
  const level = renderer.getZoomLevel();
  zoomLevelDisplay.textContent = `${Math.round(level * 100)}%`;
}

/**
 * Render the current visualization state.
 */
function render(): void {
  const posArray = samplerState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;
  const trueParams = {
    modes: currentDistribution.trueParams.modes,
    mean: currentDistribution.trueParams.mean,
  };
  const showTrueParams = showTrueParamsCheckbox.checked;
  renderer.render(contours, samples, x, y, prevX, prevY, trueParams, showTrueParams);
}

/**
 * Animation loop tick.
 */
function tick(): void {
  if (!isPlaying) return;

  try {
    performStep();
  } catch (error) {
    console.error('[HMC-VIZ] tick() error:', error);
    isPlaying = false;
    playPauseBtn.textContent = 'Play';
    return;
  }

  const speed = parseInt(speedSlider.value, 10);
  animationTimer = window.setTimeout(tick, speed);
}

/**
 * Toggle play/pause.
 */
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

/**
 * Set up UI event listeners.
 */
function setupEventListeners(): void {
  // Algorithm change
  algorithmSelect.addEventListener('change', () => {
    currentAlgorithm = algorithmSelect.value as Algorithm;
    reset();
  });

  // Distribution change
  distributionSelect.addEventListener('change', () => {
    const key = distributionSelect.value as keyof typeof distributions;
    currentDistribution = distributions[key]!();
    reset();
  });

  // Slider updates
  stepSizeSlider.addEventListener('input', () => {
    stepSizeValue.textContent = stepSizeSlider.value;
  });

  stepSizeSlider.addEventListener('change', () => {
    buildSampler();
    initializeState();
  });

  numStepsSlider.addEventListener('input', () => {
    numStepsValue.textContent = numStepsSlider.value;
  });

  numStepsSlider.addEventListener('change', () => {
    buildSampler();
    initializeState();
  });

  speedSlider.addEventListener('input', () => {
    speedValue.textContent = `${speedSlider.value}ms`;
  });

  // Buttons
  playPauseBtn.addEventListener('click', togglePlay);
  stepBtn.addEventListener('click', performStep);
  resetBtn.addEventListener('click', reset);

  // True params toggle
  showTrueParamsCheckbox.addEventListener('change', () => {
    render();
  });

  // Zoom controls
  zoomInBtn.addEventListener('click', () => {
    renderer.zoomIn();
    updateZoomDisplay();
    render();
  });

  zoomOutBtn.addEventListener('click', () => {
    renderer.zoomOut();
    updateZoomDisplay();
    render();
  });

  zoomResetBtn.addEventListener('click', () => {
    renderer.resetView();
    updateZoomDisplay();
    render();
  });

  // Mouse wheel zoom on canvas
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      renderer.zoomIn(1.1);
    } else {
      renderer.zoomOut(1.1);
    }
    updateZoomDisplay();
    render();
  }, { passive: false });

  // Resize handler
  window.addEventListener('resize', () => {
    renderer.resize();
    render();
  });
}

/**
 * Initialize the visualization.
 */
async function init(): Promise<void> {
  console.log('[HMC-VIZ] 1. init() starting');
  loadingEl.querySelector('.loading-text')!.textContent = 'Initializing JAX-JS...';

  // Small delay to let UI render
  await new Promise((r) => setTimeout(r, 100));

  try {
    // Initialize JAX-JS with WebGPU (fall back to wasm if unavailable)
    console.log('[HMC-VIZ] 1a. Initializing JAX-JS backends...');
    const availableDevices = await jaxInit();
    console.log('[HMC-VIZ] 1b. Available devices:', availableDevices);

    if (availableDevices.includes('webgpu')) {
      defaultDevice('webgpu');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WebGPU backend...';
      console.log('[HMC-VIZ] 1c. Using WebGPU backend');
    } else {
      console.warn('[HMC-VIZ] 1c. WebGPU not available, using wasm backend');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WASM backend (WebGPU unavailable)...';
    }

    await new Promise((r) => setTimeout(r, 50));

    // Initialize algorithm and distribution
    currentAlgorithm = algorithmSelect.value as Algorithm;
    currentDistribution = distributions.gaussian!();
    console.log('[HMC-VIZ] 2. Distribution created:', currentDistribution.name);
    trueParamsInfo.textContent = currentDistribution.trueParams.description;

    // Create renderer
    renderer = new CanvasRenderer(canvas, currentDistribution.bounds);
    console.log('[HMC-VIZ] 3. Renderer created');

    // Build sampler and state (this triggers JAX-JS warmup)
    loadingEl.querySelector('.loading-text')!.textContent = 'Building sampler...';
    await new Promise((r) => setTimeout(r, 50));

    console.log('[HMC-VIZ] 4. Building sampler...');
    syncAlgorithmUI();
    buildSampler();
    console.log('[HMC-VIZ] 5. Sampler built');
    initializeState();
    console.log('[HMC-VIZ] 6. State initialized');

    // Compute contours
    loadingEl.querySelector('.loading-text')!.textContent = 'Computing contours...';
    await new Promise((r) => setTimeout(r, 50));
    console.log('[HMC-VIZ] 7. Computing contours...');
    computeContours();
    console.log('[HMC-VIZ] 8. Contours computed');

    // Initial render
    console.log('[HMC-VIZ] 9. Rendering...');
    render();

    // Setup UI
    setupEventListeners();

    // Hide loading
    loadingEl.style.display = 'none';
    console.log('[HMC-VIZ] 10. Complete!');
  } catch (error) {
    loadingEl.querySelector('.loading-text')!.textContent =
      `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    console.error('[HMC-VIZ] Initialization error:', error);
  }
}

// Debug API types for agentic debugging
interface DebugState {
  algorithm: Algorithm;
  distribution: string;
  position: [number, number];
  stepCount: number;
  acceptedCount: number;
  divergentCount: number;
  acceptanceRate: number;
  config: { stepSize: number; numIntegrationSteps: number };
  recentSamples: Array<{ x: number; y: number; accepted: boolean; divergent: boolean }>;
}

interface DebugStepResult {
  accepted: boolean;
  acceptanceProb: number;
  position: [number, number];
  energy?: number;
  isDivergent?: boolean;
}

interface DebugConfig {
  algorithm?: Algorithm;
  stepSize?: number;
  numSteps?: number;
  distribution?: string;
}

// Last step info for debug API
let lastStepResult: DebugStepResult | null = null;

// Expose control interface to window for API control
interface HMCVizAPI {
  play: () => void;
  pause: () => void;
  step: () => { isAccepted: boolean; isDivergent: boolean; samples: number } | null;
  reset: () => void;
  getStatus: () => {
    isPlaying: boolean;
    samples: number;
    stepCounter: number;
    acceptedCount: number;
    divergentCount: number;
    acceptanceRate: number;
    currentDistribution: string;
  };
  getDistributions: () => string[];
  setDistribution: (name: string) => { success: boolean; error?: string; distribution?: string };
  setStepSize: (size: number) => { success: boolean; stepSize: number };
  setNumSteps: (steps: number) => { success: boolean; numSteps: number };
}

(window as unknown as { __hmcViz: HMCVizAPI }).__hmcViz = {
  play: () => {
    if (!isPlaying) {
      togglePlay();
    }
  },
  pause: () => {
    if (isPlaying) {
      togglePlay();
    }
  },
  step: () => {
    if (isPlaying) {
      // Pause first if playing
      togglePlay();
    }
    try {
      performStep();
      const lastSample = samples[samples.length - 1];
      return lastSample
        ? { isAccepted: lastSample.accepted, isDivergent: lastSample.divergent, samples: samples.length }
        : null;
    } catch (error) {
      console.error('[HMC-VIZ] step() error:', error);
      return null;
    }
  },
  reset: () => {
    reset();
  },
  getStatus: () => {
    const posArray = samplerState.position.ref.js() as number[];
    return {
      isPlaying,
      samples: samples.length,
      stepCounter,
      acceptedCount,
      divergentCount,
      acceptanceRate: samples.length > 0 ? acceptedCount / samples.length : 0,
      currentDistribution: currentDistribution.name,
      position: { x: posArray[0], y: posArray[1] },
    };
  },
  getDistributions: () => Object.keys(distributions),
  setDistribution: (name: string) => {
    if (!(name in distributions)) {
      return { success: false, error: `Unknown distribution: ${name}. Available: ${Object.keys(distributions).join(', ')}` };
    }
    try {
      currentDistribution = distributions[name as keyof typeof distributions]!();
      distributionSelect.value = name;
      reset();
      console.log(`[HMC-VIZ] Distribution changed to: ${currentDistribution.name}`);
      return { success: true, distribution: currentDistribution.name };
    } catch (error) {
      console.error('[HMC-VIZ] setDistribution error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  },
  setStepSize: (size: number) => {
    stepSizeSlider.value = String(size);
    stepSizeValue.textContent = String(size);
    buildSampler();
    initializeState();
    console.log(`[HMC-VIZ] Step size changed to: ${size}`);
    return { success: true, stepSize: size };
  },
  setNumSteps: (steps: number) => {
    numStepsSlider.value = String(steps);
    numStepsValue.textContent = String(steps);
    buildSampler();
    initializeState();
    console.log(`[HMC-VIZ] Num steps changed to: ${steps}`);
    return { success: true, numSteps: steps };
  },
};

console.log('[HMC-VIZ] API exposed to window.__hmcViz');

// Debug API for agentic debugging via command queue pattern
interface HMCDebugAPI {
  getState: () => DebugState;
  step: () => DebugStepResult;
  reset: () => { ok: true; position: [number, number] };
  setConfig: (config: DebugConfig) => { stepSize: number; numIntegrationSteps: number; algorithm: Algorithm; distribution: string };
}

(window as unknown as { __hmcDebug: HMCDebugAPI }).__hmcDebug = {
  getState: (): DebugState => {
    const posArray = samplerState.position.ref.js() as number[];
    return {
      algorithm: currentAlgorithm,
      distribution: currentDistribution.name,
      position: [posArray[0]!, posArray[1]!],
      stepCount: stepCounter,
      acceptedCount,
      divergentCount,
      acceptanceRate: stepCounter > 0 ? acceptedCount / stepCounter : 0,
      config: {
        stepSize: parseFloat(stepSizeSlider.value),
        numIntegrationSteps: parseInt(numStepsSlider.value, 10),
      },
      recentSamples: samples.slice(-20),
    };
  },
  step: (): DebugStepResult => {
    if (isPlaying) {
      togglePlay(); // Pause first
    }
    performStep();
    return lastStepResult!;
  },
  reset: (): { ok: true; position: [number, number] } => {
    reset();
    return { ok: true, position: currentDistribution.initialPosition as [number, number] };
  },
  setConfig: (config: DebugConfig) => {
    if (config.algorithm !== undefined && config.algorithm !== currentAlgorithm) {
      currentAlgorithm = config.algorithm;
      algorithmSelect.value = config.algorithm;
      syncAlgorithmUI();
    }
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
      stepSizeValue.textContent = String(config.stepSize);
    }
    if (config.numSteps !== undefined) {
      numStepsSlider.value = String(config.numSteps);
      numStepsValue.textContent = String(config.numSteps);
    }
    buildSampler();
    initializeState();
    computeContours();
    renderer.setBounds(currentDistribution.bounds);
    render();
    console.log('[HMC-VIZ] Config updated:', config);
    return {
      stepSize: parseFloat(stepSizeSlider.value),
      numIntegrationSteps: parseInt(numStepsSlider.value, 10),
      algorithm: currentAlgorithm,
      distribution: currentDistribution.name,
    };
  },
};

console.log('[HMC-VIZ] Debug API exposed to window.__hmcDebug');

// Debug command polling loop (only in dev mode)
async function debugPollLoop(): Promise<void> {
  if (import.meta.env.PROD) return;

  while (true) {
    try {
      const res = await fetch('/__debug/poll');
      const cmd = await res.json() as { id: string; type: string; payload?: unknown } | null;

      if (cmd?.type) {
        const api = (window as unknown as { __hmcDebug: HMCDebugAPI }).__hmcDebug;
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

// Set up HMR WebSocket handler for API control
if (import.meta.hot) {
  import.meta.hot.on('hmcviz:command', (data: { requestId: string; command: string }) => {
    const api = (window as unknown as { __hmcViz: HMCVizAPI }).__hmcViz;
    let result: unknown;

    try {
      switch (data.command) {
        case 'getStatus':
          result = api.getStatus();
          break;
        case 'play':
          api.play();
          result = api.getStatus();
          break;
        case 'pause':
          api.pause();
          result = api.getStatus();
          break;
        case 'step':
          result = api.step();
          break;
        case 'reset':
          api.reset();
          result = api.getStatus();
          break;
        case 'getDistributions':
          result = api.getDistributions();
          break;
        default:
          // Handle setDistribution:name command
          if (data.command.startsWith('setDistribution:')) {
            const distName = data.command.replace('setDistribution:', '');
            result = api.setDistribution(distName);
          } else if (data.command.startsWith('setStepSize:')) {
            const size = parseFloat(data.command.replace('setStepSize:', ''));
            result = api.setStepSize(size);
          } else if (data.command.startsWith('setNumSteps:')) {
            const steps = parseInt(data.command.replace('setNumSteps:', ''), 10);
            result = api.setNumSteps(steps);
          } else {
            result = { error: 'Unknown command' };
          }
      }
    } catch (error) {
      console.error('[HMC-VIZ] Command error:', error);
      result = { error: error instanceof Error ? error.message : 'Unknown error' };
    }

    import.meta.hot!.send('hmcviz:response', { requestId: data.requestId, result });
  });
  console.log('[HMC-VIZ] HMR command handler registered');
}

// Start
init();
