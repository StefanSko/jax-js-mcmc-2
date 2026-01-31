/**
 * HMC Explorer Visualization
 *
 * Capstone page: HMC trajectories with an optional side-by-side RWM comparison.
 * Demonstrates why HMC's geometry-informed proposals explore faster than RWM's
 * random-walk proposals on the same target distribution.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

import { numpy as np, random, init as jaxInit, defaultDevice } from '@jax-js/jax';
import { HMC, RWM, type HMCInfo, type HMCState, type RWMInfo, type RWMState } from '../../../src';
import { distributions, type Distribution } from '../distributions';
import { computeDensityGrid, computeContourLevels, extractContours, type ContourLine } from '../contour';
import { CanvasRenderer, type Sample } from '../renderer';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Mode = 'single' | 'comparison';

type HMCSampler = {
  init: (position: np.Array) => HMCState;
  step: (key: np.Array, state: HMCState) => [HMCState, HMCInfo];
};

type RWMSamplerType = {
  init: (position: np.Array) => RWMState;
  step: (key: np.Array, state: RWMState) => [RWMState, RWMInfo];
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentMode: Mode = 'single';

// HMC state
let hmcSampler: HMCSampler;
let hmcState: HMCState;
let hmcSamples: Sample[] = [];
let hmcStepCounter = 0;
let hmcAcceptedCount = 0;
let hmcPrevX: number | undefined;
let hmcPrevY: number | undefined;

// RWM state (used in comparison mode)
let rwmSampler: RWMSamplerType;
let rwmState: RWMState;
let rwmSamples: Sample[] = [];
let rwmStepCounter = 0;
let rwmAcceptedCount = 0;
let rwmPrevX: number | undefined;
let rwmPrevY: number | undefined;

// Shared
let currentDistribution: Distribution;
let contours: ContourLine[] = [];
let hmcRenderer: CanvasRenderer;
let rwmRenderer: CanvasRenderer;
let showTrajectory = true;
let isPlaying = false;
let animationTimer: number | null = null;
let baseSeed = Date.now();
let rwmBuilt = false;

// Last step results for debug API
let lastHMCStepResult: {
  accepted: boolean;
  acceptanceProb: number;
  position: [number, number];
} | null = null;

let lastRWMStepResult: {
  accepted: boolean;
  acceptanceProb: number;
  position: [number, number];
} | null = null;

// ---------------------------------------------------------------------------
// UI Elements - Single mode
// ---------------------------------------------------------------------------

const singleLayout = document.getElementById('single-layout') as HTMLDivElement;
const comparisonLayout = document.getElementById('comparison-layout') as HTMLDivElement;

const hmcCanvas = document.getElementById('hmc-canvas') as HTMLCanvasElement;
const rwmCanvas = document.getElementById('rwm-canvas') as HTMLCanvasElement;
const hmcCanvasCmp = document.getElementById('hmc-canvas-cmp') as HTMLCanvasElement;
const loadingEl = document.getElementById('loading') as HTMLDivElement;

// Single mode controls
const modeSingleBtn = document.getElementById('mode-single-btn') as HTMLButtonElement;
const modeComparisonBtn = document.getElementById('mode-comparison-btn') as HTMLButtonElement;
const distributionSelect = document.getElementById('distribution') as HTMLSelectElement;
const stepSizeSlider = document.getElementById('step-size') as HTMLInputElement;
const stepSizeValue = document.getElementById('step-size-value') as HTMLSpanElement;
const numStepsSlider = document.getElementById('num-steps') as HTMLInputElement;
const numStepsValue = document.getElementById('num-steps-value') as HTMLSpanElement;
const showTrajectoryCheckbox = document.getElementById('show-trajectory') as HTMLInputElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const speedValue = document.getElementById('speed-value') as HTMLSpanElement;
const playPauseBtn = document.getElementById('play-pause') as HTMLButtonElement;
const stepBtn = document.getElementById('step-btn') as HTMLButtonElement;
const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

// Single mode HMC stats
const hmcStatSamples = document.getElementById('hmc-stat-samples') as HTMLSpanElement;
const hmcStatAcceptance = document.getElementById('hmc-stat-acceptance') as HTMLSpanElement;
const hmcStatMeanX = document.getElementById('hmc-stat-mean-x') as HTMLSpanElement;
const hmcStatMeanY = document.getElementById('hmc-stat-mean-y') as HTMLSpanElement;

// Single mode RWM stats (shown when in comparison)
const rwmStatsSingleGroup = document.getElementById('rwm-stats-single-group') as HTMLDivElement;
const rwmStatSamples = document.getElementById('rwm-stat-samples') as HTMLSpanElement;
const rwmStatAcceptance = document.getElementById('rwm-stat-acceptance') as HTMLSpanElement;
const rwmStatMeanX = document.getElementById('rwm-stat-mean-x') as HTMLSpanElement;
const rwmStatMeanY = document.getElementById('rwm-stat-mean-y') as HTMLSpanElement;

// Comparison mode controls
const modeSingleBtnCmp = document.getElementById('mode-single-btn-cmp') as HTMLButtonElement;
const modeComparisonBtnCmp = document.getElementById('mode-comparison-btn-cmp') as HTMLButtonElement;
const distributionSelectCmp = document.getElementById('distribution-cmp') as HTMLSelectElement;
const stepSizeSliderCmp = document.getElementById('step-size-cmp') as HTMLInputElement;
const stepSizeValueCmp = document.getElementById('step-size-value-cmp') as HTMLSpanElement;
const numStepsSliderCmp = document.getElementById('num-steps-cmp') as HTMLInputElement;
const numStepsValueCmp = document.getElementById('num-steps-value-cmp') as HTMLSpanElement;
const showTrajectoryCmpCheckbox = document.getElementById('show-trajectory-cmp') as HTMLInputElement;
const speedSliderCmp = document.getElementById('speed-cmp') as HTMLInputElement;
const speedValueCmp = document.getElementById('speed-value-cmp') as HTMLSpanElement;
const playPauseBtnCmp = document.getElementById('play-pause-cmp') as HTMLButtonElement;
const stepBtnCmp = document.getElementById('step-btn-cmp') as HTMLButtonElement;
const resetBtnCmp = document.getElementById('reset-btn-cmp') as HTMLButtonElement;

// Comparison mode stats
const rwmStatSamplesCmp = document.getElementById('rwm-stat-samples-cmp') as HTMLSpanElement;
const rwmStatAcceptanceCmp = document.getElementById('rwm-stat-acceptance-cmp') as HTMLSpanElement;
const rwmStatMeanXCmp = document.getElementById('rwm-stat-mean-x-cmp') as HTMLSpanElement;
const rwmStatMeanYCmp = document.getElementById('rwm-stat-mean-y-cmp') as HTMLSpanElement;
const hmcStatSamplesCmp = document.getElementById('hmc-stat-samples-cmp') as HTMLSpanElement;
const hmcStatAcceptanceCmp = document.getElementById('hmc-stat-acceptance-cmp') as HTMLSpanElement;
const hmcStatMeanXCmp = document.getElementById('hmc-stat-mean-x-cmp') as HTMLSpanElement;
const hmcStatMeanYCmp = document.getElementById('hmc-stat-mean-y-cmp') as HTMLSpanElement;

// ---------------------------------------------------------------------------
// Pure JS log-density evaluators for contour computation
// ---------------------------------------------------------------------------

function createJSLogdensity(dist: Distribution): (x: number, y: number) => number {
  const name = dist.name;

  if (name === '2D Gaussian') {
    return (x: number, y: number) => -0.5 * (x * x + y * y);
  }

  if (name === 'Correlated Gaussian') {
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
// Info disposal helpers
// ---------------------------------------------------------------------------

function disposeHMCInfo(info: HMCInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.momentum.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();
}

function disposeRWMInfo(info: RWMInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.proposedPosition.dispose();
}

// ---------------------------------------------------------------------------
// Sampler builders
// ---------------------------------------------------------------------------

function getStepSize(): number {
  return parseFloat(
    currentMode === 'single' ? stepSizeSlider.value : stepSizeSliderCmp.value
  );
}

function getNumSteps(): number {
  return parseInt(
    currentMode === 'single' ? numStepsSlider.value : numStepsSliderCmp.value,
    10
  );
}

function getSpeed(): number {
  return parseInt(
    currentMode === 'single' ? speedSlider.value : speedSliderCmp.value,
    10
  );
}

function buildHMCSampler(): void {
  const stepSize = getStepSize();
  const numSteps = getNumSteps();
  console.log('[HMC-EXPLORER] buildHMCSampler:', { stepSize, numSteps });

  hmcSampler = HMC(currentDistribution.logdensity)
    .stepSize(stepSize)
    .numIntegrationSteps(numSteps)
    .inverseMassMatrix(np.array([1.0, 1.0]))
    .build();
}

function buildRWMSampler(): void {
  // Use HMC step size * 2 for RWM to give it a fair chance
  const stepSize = getStepSize() * 2;
  console.log('[HMC-EXPLORER] buildRWMSampler:', { stepSize });

  rwmSampler = RWM(currentDistribution.logdensity)
    .stepSize(stepSize)
    .build();

  rwmBuilt = true;
}

// ---------------------------------------------------------------------------
// State initialization
// ---------------------------------------------------------------------------

function initializeHMCState(): void {
  const [x, y] = currentDistribution.initialPosition;
  hmcState = hmcSampler.init(np.array([x, y]));
  hmcPrevX = undefined;
  hmcPrevY = undefined;
}

function initializeRWMState(): void {
  const [x, y] = currentDistribution.initialPosition;
  rwmState = rwmSampler.init(np.array([x, y]));
  rwmPrevX = undefined;
  rwmPrevY = undefined;
}

// ---------------------------------------------------------------------------
// Perform sampler steps
// ---------------------------------------------------------------------------

function performHMCStep(): { accepted: boolean; acceptanceProb: number; position: [number, number] } {
  // Save old position BEFORE step (step consumes the state)
  const oldPos = hmcState.position.ref.js() as number[];
  hmcPrevX = oldPos[0];
  hmcPrevY = oldPos[1];

  const key = random.key(baseSeed + hmcStepCounter);
  const [newState, info] = hmcSampler.step(key, hmcState);

  // Read info values before disposal
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;

  // Capture for debug API
  const newPosArray = newState.position.ref.js() as number[];
  lastHMCStepResult = {
    accepted: isAccepted,
    acceptanceProb,
    position: [newPosArray[0]!, newPosArray[1]!],
  };

  // Debug logging
  console.log(`[HMC-EXPLORER] HMC Step ${hmcStepCounter}: pos=(${hmcPrevX?.toFixed(2)}, ${hmcPrevY?.toFixed(2)}) -> acceptProb=${acceptanceProb.toFixed(4)}, accepted=${isAccepted}`);

  // Dispose info (old state was consumed by step)
  disposeHMCInfo(info);

  // Update state
  hmcState = newState;
  hmcStepCounter++;

  // Get new position
  const posArray = newState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;

  // Track sample
  hmcSamples.push({ x, y, accepted: isAccepted });
  if (isAccepted) hmcAcceptedCount++;

  // Limit samples to prevent memory issues
  if (hmcSamples.length > 500) {
    hmcSamples = hmcSamples.slice(-500);
  }

  return { accepted: isAccepted, acceptanceProb, position: [x, y] };
}

function performRWMStep(): { accepted: boolean; acceptanceProb: number; position: [number, number] } {
  // Save old position BEFORE step (step consumes the state)
  const oldPos = rwmState.position.ref.js() as number[];
  rwmPrevX = oldPos[0];
  rwmPrevY = oldPos[1];

  // Offset seed for RWM to get different randomness from HMC
  const key = random.key(baseSeed + rwmStepCounter + 1000000);
  const [newState, info] = rwmSampler.step(key, rwmState);

  // Read info values before disposal
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;

  // Capture for debug API
  const newPosArray = newState.position.ref.js() as number[];
  lastRWMStepResult = {
    accepted: isAccepted,
    acceptanceProb,
    position: [newPosArray[0]!, newPosArray[1]!],
  };

  // Debug logging
  console.log(`[HMC-EXPLORER] RWM Step ${rwmStepCounter}: pos=(${rwmPrevX?.toFixed(2)}, ${rwmPrevY?.toFixed(2)}) -> acceptProb=${acceptanceProb.toFixed(4)}, accepted=${isAccepted}`);

  // Dispose info (old state was consumed by step)
  disposeRWMInfo(info);

  // Update state
  rwmState = newState;
  rwmStepCounter++;

  // Get new position
  const posArray = newState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;

  // Track sample
  rwmSamples.push({ x, y, accepted: isAccepted });
  if (isAccepted) rwmAcceptedCount++;

  // Limit samples to prevent memory issues
  if (rwmSamples.length > 500) {
    rwmSamples = rwmSamples.slice(-500);
  }

  return { accepted: isAccepted, acceptanceProb, position: [x, y] };
}

function performStep(): void {
  // Always run HMC step
  performHMCStep();

  // In comparison mode, also run RWM step
  if (currentMode === 'comparison' && rwmBuilt) {
    performRWMStep();
  }

  // Update UI
  updateStats();
  render();
}

// ---------------------------------------------------------------------------
// Statistics display
// ---------------------------------------------------------------------------

function updateHMCStats(
  samplesEl: HTMLSpanElement,
  acceptanceEl: HTMLSpanElement,
  meanXEl: HTMLSpanElement,
  meanYEl: HTMLSpanElement,
): void {
  samplesEl.textContent = String(hmcStepCounter);

  if (hmcStepCounter > 0) {
    const rate = (hmcAcceptedCount / hmcStepCounter) * 100;
    acceptanceEl.textContent = `${rate.toFixed(1)}%`;

    if (hmcSamples.length > 0) {
      const meanX = hmcSamples.reduce((sum, s) => sum + s.x, 0) / hmcSamples.length;
      const meanY = hmcSamples.reduce((sum, s) => sum + s.y, 0) / hmcSamples.length;
      meanXEl.textContent = meanX.toFixed(3);
      meanYEl.textContent = meanY.toFixed(3);
    }
  } else {
    acceptanceEl.textContent = '-';
    meanXEl.textContent = '-';
    meanYEl.textContent = '-';
  }
}

function updateRWMStats(
  samplesEl: HTMLSpanElement,
  acceptanceEl: HTMLSpanElement,
  meanXEl: HTMLSpanElement,
  meanYEl: HTMLSpanElement,
): void {
  samplesEl.textContent = String(rwmStepCounter);

  if (rwmStepCounter > 0) {
    const rate = (rwmAcceptedCount / rwmStepCounter) * 100;
    acceptanceEl.textContent = `${rate.toFixed(1)}%`;

    if (rwmSamples.length > 0) {
      const meanX = rwmSamples.reduce((sum, s) => sum + s.x, 0) / rwmSamples.length;
      const meanY = rwmSamples.reduce((sum, s) => sum + s.y, 0) / rwmSamples.length;
      meanXEl.textContent = meanX.toFixed(3);
      meanYEl.textContent = meanY.toFixed(3);
    }
  } else {
    acceptanceEl.textContent = '-';
    meanXEl.textContent = '-';
    meanYEl.textContent = '-';
  }
}

function updateStats(): void {
  if (currentMode === 'single') {
    updateHMCStats(hmcStatSamples, hmcStatAcceptance, hmcStatMeanX, hmcStatMeanY);
  } else {
    // Comparison mode: update both sets of stats
    updateHMCStats(hmcStatSamplesCmp, hmcStatAcceptanceCmp, hmcStatMeanXCmp, hmcStatMeanYCmp);
    updateRWMStats(rwmStatSamplesCmp, rwmStatAcceptanceCmp, rwmStatMeanXCmp, rwmStatMeanYCmp);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderHMCCanvas(renderer: CanvasRenderer): void {
  const posArray = hmcState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;

  const prevX = showTrajectory ? hmcPrevX : undefined;
  const prevY = showTrajectory ? hmcPrevY : undefined;

  renderer.render(contours, hmcSamples, x, y, prevX, prevY);
}

function renderRWMCanvas(): void {
  if (!rwmBuilt || !rwmState) return;

  const posArray = rwmState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;

  const prevX = showTrajectory ? rwmPrevX : undefined;
  const prevY = showTrajectory ? rwmPrevY : undefined;

  rwmRenderer.render(contours, rwmSamples, x, y, prevX, prevY);
}

function render(): void {
  if (currentMode === 'single') {
    renderHMCCanvas(hmcRenderer);
  } else {
    // Comparison mode: render both canvases
    // HMC uses the comparison canvas
    renderHMCCanvas(hmcRenderer);
    renderRWMCanvas();
  }
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

function switchToSingleMode(): void {
  if (currentMode === 'single') return;
  currentMode = 'single';

  console.log('[HMC-EXPLORER] Switching to single mode');

  // Stop animation
  stopAnimation();

  // Sync control values from comparison to single
  syncControlValues('comparison-to-single');

  // Switch layouts
  singleLayout.style.display = 'flex';
  comparisonLayout.style.display = 'none';
  rwmCanvas.style.display = 'none';
  rwmStatsSingleGroup.classList.add('hidden');

  // Update mode toggle buttons
  modeSingleBtn.classList.add('active');
  modeComparisonBtn.classList.remove('active');

  // Rebuild HMC renderer on the single canvas
  hmcRenderer = new CanvasRenderer(hmcCanvas, currentDistribution.bounds);

  // Need a small delay for canvas to be visible and sized
  requestAnimationFrame(() => {
    hmcRenderer.resize();
    render();
  });
}

function switchToComparisonMode(): void {
  if (currentMode === 'comparison') return;
  currentMode = 'comparison';

  console.log('[HMC-EXPLORER] Switching to comparison mode');

  // Stop animation
  stopAnimation();

  // Sync control values from single to comparison
  syncControlValues('single-to-comparison');

  // Switch layouts
  singleLayout.style.display = 'none';
  comparisonLayout.style.display = 'flex';
  rwmCanvas.style.display = 'block';

  // Update mode toggle buttons
  modeComparisonBtnCmp.classList.add('active');
  modeSingleBtnCmp.classList.remove('active');

  // Build RWM sampler if not already built
  if (!rwmBuilt) {
    buildRWMSampler();
    // Initialize RWM state at same position as HMC initial position
    initializeRWMState();
  }

  // Rebuild renderers on comparison canvases
  hmcRenderer = new CanvasRenderer(hmcCanvasCmp, currentDistribution.bounds);
  rwmRenderer = new CanvasRenderer(rwmCanvas, currentDistribution.bounds);

  // Need a small delay for canvases to be visible and sized
  requestAnimationFrame(() => {
    hmcRenderer.resize();
    rwmRenderer.resize();
    updateStats();
    render();
  });
}

function syncControlValues(direction: 'single-to-comparison' | 'comparison-to-single'): void {
  if (direction === 'single-to-comparison') {
    stepSizeSliderCmp.value = stepSizeSlider.value;
    stepSizeValueCmp.textContent = stepSizeSlider.value;
    numStepsSliderCmp.value = numStepsSlider.value;
    numStepsValueCmp.textContent = numStepsSlider.value;
    showTrajectoryCmpCheckbox.checked = showTrajectoryCheckbox.checked;
    speedSliderCmp.value = speedSlider.value;
    speedValueCmp.textContent = `${speedSlider.value}ms`;
    distributionSelectCmp.value = distributionSelect.value;
  } else {
    stepSizeSlider.value = stepSizeSliderCmp.value;
    stepSizeValue.textContent = stepSizeSliderCmp.value;
    numStepsSlider.value = numStepsSliderCmp.value;
    numStepsValue.textContent = numStepsSliderCmp.value;
    showTrajectoryCheckbox.checked = showTrajectoryCmpCheckbox.checked;
    speedSlider.value = speedSliderCmp.value;
    speedValue.textContent = `${speedSliderCmp.value}ms`;
    distributionSelect.value = distributionSelectCmp.value;
  }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

function reset(): void {
  // Stop animation
  stopAnimation();

  // Reset stats
  hmcSamples = [];
  hmcStepCounter = 0;
  hmcAcceptedCount = 0;
  hmcPrevX = undefined;
  hmcPrevY = undefined;
  lastHMCStepResult = null;

  rwmSamples = [];
  rwmStepCounter = 0;
  rwmAcceptedCount = 0;
  rwmPrevX = undefined;
  rwmPrevY = undefined;
  lastRWMStepResult = null;

  baseSeed = Date.now();

  // Rebuild samplers and state
  buildHMCSampler();
  initializeHMCState();

  if (currentMode === 'comparison') {
    buildRWMSampler();
    initializeRWMState();
    rwmRenderer.setBounds(currentDistribution.bounds);
  } else {
    rwmBuilt = false;
  }

  computeContours();
  hmcRenderer.setBounds(currentDistribution.bounds);

  // Update UI
  updateStats();
  render();
}

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------

function stopAnimation(): void {
  if (animationTimer !== null) {
    clearTimeout(animationTimer);
    animationTimer = null;
  }
  isPlaying = false;
  playPauseBtn.textContent = 'Play';
  playPauseBtnCmp.textContent = 'Play';
}

function tick(): void {
  if (!isPlaying) return;

  try {
    performStep();
  } catch (error) {
    console.error('[HMC-EXPLORER] tick() error:', error);
    stopAnimation();
    return;
  }

  const speed = getSpeed();
  animationTimer = window.setTimeout(tick, speed);
}

function togglePlay(): void {
  isPlaying = !isPlaying;
  const label = isPlaying ? 'Pause' : 'Play';
  playPauseBtn.textContent = label;
  playPauseBtnCmp.textContent = label;

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
  // --- Mode toggle (single controls) ---
  modeSingleBtn.addEventListener('click', switchToSingleMode);
  modeComparisonBtn.addEventListener('click', switchToComparisonMode);

  // --- Mode toggle (comparison controls) ---
  modeSingleBtnCmp.addEventListener('click', switchToSingleMode);
  modeComparisonBtnCmp.addEventListener('click', switchToComparisonMode);

  // --- Distribution change ---
  distributionSelect.addEventListener('change', () => {
    const key = distributionSelect.value as keyof typeof distributions;
    currentDistribution = distributions[key]!();
    distributionSelectCmp.value = distributionSelect.value;
    reset();
  });

  distributionSelectCmp.addEventListener('change', () => {
    const key = distributionSelectCmp.value as keyof typeof distributions;
    currentDistribution = distributions[key]!();
    distributionSelect.value = distributionSelectCmp.value;
    reset();
  });

  // --- Step size sliders ---
  stepSizeSlider.addEventListener('input', () => {
    stepSizeValue.textContent = stepSizeSlider.value;
  });

  stepSizeSlider.addEventListener('change', () => {
    stepSizeSliderCmp.value = stepSizeSlider.value;
    stepSizeValueCmp.textContent = stepSizeSlider.value;
    rebuildSamplers();
  });

  stepSizeSliderCmp.addEventListener('input', () => {
    stepSizeValueCmp.textContent = stepSizeSliderCmp.value;
  });

  stepSizeSliderCmp.addEventListener('change', () => {
    stepSizeSlider.value = stepSizeSliderCmp.value;
    stepSizeValue.textContent = stepSizeSliderCmp.value;
    rebuildSamplers();
  });

  // --- Num steps sliders ---
  numStepsSlider.addEventListener('input', () => {
    numStepsValue.textContent = numStepsSlider.value;
  });

  numStepsSlider.addEventListener('change', () => {
    numStepsSliderCmp.value = numStepsSlider.value;
    numStepsValueCmp.textContent = numStepsSlider.value;
    rebuildHMCSampler();
  });

  numStepsSliderCmp.addEventListener('input', () => {
    numStepsValueCmp.textContent = numStepsSliderCmp.value;
  });

  numStepsSliderCmp.addEventListener('change', () => {
    numStepsSlider.value = numStepsSliderCmp.value;
    numStepsValue.textContent = numStepsSliderCmp.value;
    rebuildHMCSampler();
  });

  // --- Show trajectory checkboxes ---
  showTrajectoryCheckbox.addEventListener('change', () => {
    showTrajectory = showTrajectoryCheckbox.checked;
    showTrajectoryCmpCheckbox.checked = showTrajectory;
    render();
  });

  showTrajectoryCmpCheckbox.addEventListener('change', () => {
    showTrajectory = showTrajectoryCmpCheckbox.checked;
    showTrajectoryCheckbox.checked = showTrajectory;
    render();
  });

  // --- Speed sliders ---
  speedSlider.addEventListener('input', () => {
    speedValue.textContent = `${speedSlider.value}ms`;
  });

  speedSliderCmp.addEventListener('input', () => {
    speedValueCmp.textContent = `${speedSliderCmp.value}ms`;
  });

  // --- Action buttons (single) ---
  playPauseBtn.addEventListener('click', togglePlay);
  stepBtn.addEventListener('click', performStep);
  resetBtn.addEventListener('click', reset);

  // --- Action buttons (comparison) ---
  playPauseBtnCmp.addEventListener('click', togglePlay);
  stepBtnCmp.addEventListener('click', performStep);
  resetBtnCmp.addEventListener('click', reset);

  // --- Window resize ---
  window.addEventListener('resize', () => {
    hmcRenderer.resize();
    if (currentMode === 'comparison' && rwmRenderer) {
      rwmRenderer.resize();
    }
    render();
  });
}

function rebuildHMCSampler(): void {
  buildHMCSampler();
  initializeHMCState();
  hmcSamples = [];
  hmcStepCounter = 0;
  hmcAcceptedCount = 0;
  updateStats();
  render();
}

function rebuildSamplers(): void {
  buildHMCSampler();
  initializeHMCState();
  hmcSamples = [];
  hmcStepCounter = 0;
  hmcAcceptedCount = 0;

  if (currentMode === 'comparison') {
    buildRWMSampler();
    initializeRWMState();
    rwmSamples = [];
    rwmStepCounter = 0;
    rwmAcceptedCount = 0;
  }

  updateStats();
  render();
}

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

interface VizDebugAPI {
  getState: () => {
    mode: Mode;
    showTrajectory: boolean;
    hmcState: {
      position: [number, number];
      stepCount: number;
      acceptanceRate: number;
      config: { stepSize: number; numSteps: number };
    };
    rwmState?: {
      position: [number, number];
      stepCount: number;
      acceptanceRate: number;
    };
  };
  step: () => {
    hmc: { accepted: boolean; acceptanceProb: number; position: [number, number] };
    rwm?: { accepted: boolean; acceptanceProb: number; position: [number, number] };
  };
  reset: () => { ok: true };
  setConfig: (config: {
    mode?: string;
    stepSize?: number;
    numSteps?: number;
    showTrajectory?: boolean;
    distribution?: string;
  }) => {
    mode: Mode;
    stepSize: number;
    numSteps: number;
    showTrajectory: boolean;
    distribution: string;
  };
}

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = {
  getState: () => {
    const hmcPosArray = hmcState.position.ref.js() as number[];
    const result: ReturnType<VizDebugAPI['getState']> = {
      mode: currentMode,
      showTrajectory,
      hmcState: {
        position: [hmcPosArray[0]!, hmcPosArray[1]!],
        stepCount: hmcStepCounter,
        acceptanceRate: hmcStepCounter > 0 ? hmcAcceptedCount / hmcStepCounter : 0,
        config: {
          stepSize: getStepSize(),
          numSteps: getNumSteps(),
        },
      },
    };

    if (currentMode === 'comparison' && rwmBuilt && rwmState) {
      const rwmPosArray = rwmState.position.ref.js() as number[];
      result.rwmState = {
        position: [rwmPosArray[0]!, rwmPosArray[1]!],
        stepCount: rwmStepCounter,
        acceptanceRate: rwmStepCounter > 0 ? rwmAcceptedCount / rwmStepCounter : 0,
      };
    }

    return result;
  },

  step: () => {
    if (isPlaying) {
      togglePlay(); // Pause first
    }

    const hmcResult = performHMCStep();
    let rwmResult: { accepted: boolean; acceptanceProb: number; position: [number, number] } | undefined;

    if (currentMode === 'comparison' && rwmBuilt) {
      rwmResult = performRWMStep();
    }

    updateStats();
    render();

    const result: ReturnType<VizDebugAPI['step']> = {
      hmc: hmcResult,
    };

    if (rwmResult) {
      result.rwm = rwmResult;
    }

    return result;
  },

  reset: () => {
    reset();
    return { ok: true as const };
  },

  setConfig: (config) => {
    if (config.mode !== undefined) {
      if (config.mode === 'comparison' && currentMode === 'single') {
        switchToComparisonMode();
      } else if (config.mode === 'single' && currentMode === 'comparison') {
        switchToSingleMode();
      }
    }

    if (config.distribution !== undefined) {
      const key = Object.keys(distributions).find(
        (k) => distributions[k as keyof typeof distributions]?.().name === config.distribution ||
               k === config.distribution
      );
      if (key) {
        currentDistribution = distributions[key as keyof typeof distributions]!();
        distributionSelect.value = key;
        distributionSelectCmp.value = key;
      }
    }

    if (config.stepSize !== undefined) {
      stepSizeSlider.value = String(config.stepSize);
      stepSizeValue.textContent = String(config.stepSize);
      stepSizeSliderCmp.value = String(config.stepSize);
      stepSizeValueCmp.textContent = String(config.stepSize);
    }

    if (config.numSteps !== undefined) {
      numStepsSlider.value = String(config.numSteps);
      numStepsValue.textContent = String(config.numSteps);
      numStepsSliderCmp.value = String(config.numSteps);
      numStepsValueCmp.textContent = String(config.numSteps);
    }

    if (config.showTrajectory !== undefined) {
      showTrajectory = config.showTrajectory;
      showTrajectoryCheckbox.checked = showTrajectory;
      showTrajectoryCmpCheckbox.checked = showTrajectory;
    }

    // Rebuild everything with new config
    buildHMCSampler();
    initializeHMCState();
    hmcSamples = [];
    hmcStepCounter = 0;
    hmcAcceptedCount = 0;

    if (currentMode === 'comparison') {
      buildRWMSampler();
      initializeRWMState();
      rwmSamples = [];
      rwmStepCounter = 0;
      rwmAcceptedCount = 0;
      rwmRenderer.setBounds(currentDistribution.bounds);
    }

    computeContours();
    hmcRenderer.setBounds(currentDistribution.bounds);
    updateStats();
    render();

    console.log('[HMC-EXPLORER] Config updated:', config);

    return {
      mode: currentMode,
      stepSize: getStepSize(),
      numSteps: getNumSteps(),
      showTrajectory,
      distribution: currentDistribution.name,
    };
  },
};

console.log('[HMC-EXPLORER] Debug API exposed to window.__vizDebug');

// ---------------------------------------------------------------------------
// Debug command polling loop (dev mode only)
// ---------------------------------------------------------------------------

type DebugConfig = {
  mode?: string;
  stepSize?: number;
  numSteps?: number;
  showTrajectory?: boolean;
  distribution?: string;
};

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
  console.log('[HMC-EXPLORER] 1. init() starting');
  loadingEl.querySelector('.loading-text')!.textContent = 'Initializing JAX-JS...';

  // Small delay to let UI render
  await new Promise((r) => setTimeout(r, 100));

  try {
    // Initialize JAX-JS with WebGPU (fall back to wasm if unavailable)
    console.log('[HMC-EXPLORER] 1a. Initializing JAX-JS backends...');
    const availableDevices = await jaxInit();
    console.log('[HMC-EXPLORER] 1b. Available devices:', availableDevices);

    if (availableDevices.includes('webgpu')) {
      defaultDevice('webgpu');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WebGPU backend...';
      console.log('[HMC-EXPLORER] 1c. Using WebGPU backend');
    } else {
      console.warn('[HMC-EXPLORER] 1c. WebGPU not available, using wasm backend');
      loadingEl.querySelector('.loading-text')!.textContent = 'Using WASM backend (WebGPU unavailable)...';
    }

    await new Promise((r) => setTimeout(r, 50));

    // Initialize distribution
    currentDistribution = distributions.gaussian!();
    console.log('[HMC-EXPLORER] 2. Distribution created:', currentDistribution.name);

    // Create HMC renderer (single mode uses the single canvas)
    hmcRenderer = new CanvasRenderer(hmcCanvas, currentDistribution.bounds);
    console.log('[HMC-EXPLORER] 3. HMC Renderer created');

    // Build HMC sampler and state
    loadingEl.querySelector('.loading-text')!.textContent = 'Building HMC sampler...';
    await new Promise((r) => setTimeout(r, 50));

    console.log('[HMC-EXPLORER] 4. Building HMC sampler...');
    buildHMCSampler();
    console.log('[HMC-EXPLORER] 5. HMC sampler built');
    initializeHMCState();
    console.log('[HMC-EXPLORER] 6. HMC state initialized');

    // Compute contours
    loadingEl.querySelector('.loading-text')!.textContent = 'Computing contours...';
    await new Promise((r) => setTimeout(r, 50));
    console.log('[HMC-EXPLORER] 7. Computing contours...');
    computeContours();
    console.log('[HMC-EXPLORER] 8. Contours computed');

    // Initial render
    console.log('[HMC-EXPLORER] 9. Rendering...');
    render();

    // Setup UI
    setupEventListeners();

    // Hide loading
    loadingEl.style.display = 'none';
    console.log('[HMC-EXPLORER] 10. Complete!');
  } catch (error) {
    loadingEl.querySelector('.loading-text')!.textContent =
      `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    console.error('[HMC-EXPLORER] Initialization error:', error);
  }
}

// Start
init();
