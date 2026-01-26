/**
 * HMC Visualization - Main Module
 *
 * Interactive browser visualization of Hamiltonian Monte Carlo sampling.
 */

// Console bridge must be first - sends browser logs to terminal
import './console-bridge';

import { numpy as np, random } from '@jax-js/jax';
import { HMC, type HMCInfo, type HMCState } from '../../src';
import { distributions, type Distribution } from './distributions';
import { computeDensityGrid, computeContourLevels, extractContours, type ContourLine } from './contour';
import { CanvasRenderer, type Sample } from './renderer';

// State
let currentDistribution: Distribution;
let sampler: ReturnType<typeof HMC>['build'] extends () => infer R ? R : never;
let hmcState: HMCState;
let samples: Sample[] = [];
let contours: ContourLine[] = [];
let renderer: CanvasRenderer;
let isPlaying = false;
let animationTimer: number | null = null;
let stepCounter = 0;
let acceptedCount = 0;
let divergentCount = 0;
let prevX: number | undefined;
let prevY: number | undefined;

// UI Elements
const canvas = document.getElementById('hmc-canvas') as HTMLCanvasElement;
const loadingEl = document.getElementById('loading') as HTMLDivElement;
const distributionSelect = document.getElementById('distribution') as HTMLSelectElement;
const stepSizeSlider = document.getElementById('step-size') as HTMLInputElement;
const stepSizeValue = document.getElementById('step-size-value') as HTMLSpanElement;
const numStepsSlider = document.getElementById('num-steps') as HTMLInputElement;
const numStepsValue = document.getElementById('num-steps-value') as HTMLSpanElement;
const speedSlider = document.getElementById('speed') as HTMLInputElement;
const speedValue = document.getElementById('speed-value') as HTMLSpanElement;
const playPauseBtn = document.getElementById('play-pause') as HTMLButtonElement;
const stepBtn = document.getElementById('step') as HTMLButtonElement;
const resetBtn = document.getElementById('reset') as HTMLButtonElement;

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
 * Dispose HMCInfo arrays to prevent memory leaks.
 */
function disposeInfo(info: HMCInfo): void {
  info.momentum.dispose();
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();
}

/**
 * Create a pure JS log-density evaluator for contour computation.
 * This avoids JAX-JS overhead for grid evaluation.
 */
function createJSLogdensity(dist: Distribution): (x: number, y: number) => number {
  const name = dist.name;

  if (name === '2D Gaussian') {
    const precisionDiag = 1.333;
    const precisionOffDiag = -0.667;
    return (x: number, y: number) => {
      const quadForm = precisionDiag * (x * x + y * y) + 2 * precisionOffDiag * x * y;
      return -0.5 * quadForm;
    };
  }

  if (name === 'Banana') {
    const a = 1.0;
    const b = 100.0;
    return (x: number, y: number) => {
      const term1 = (a - x) ** 2;
      const term2 = b * (y - x * x) ** 2;
      return -0.05 * (term1 + term2);
    };
  }

  if (name === 'Funnel') {
    return (v: number, x: number) => {
      const logPv = -v * v / 18;
      const logPxGivenV = -v / 2 - (x * x) / (2 * Math.exp(v));
      return logPv + logPxGivenV;
    };
  }

  // Fallback
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

/**
 * Build HMC sampler with current parameters.
 */
function buildSampler(): void {
  const stepSize = parseFloat(stepSizeSlider.value);
  const numSteps = parseInt(numStepsSlider.value, 10);

  sampler = HMC(currentDistribution.logdensity)
    .stepSize(stepSize)
    .numIntegrationSteps(numSteps)
    .inverseMassMatrix(np.array([1.0, 1.0]))
    .build();
}

/**
 * Initialize HMC state at distribution's initial position.
 */
function initializeState(): void {
  const [x, y] = currentDistribution.initialPosition;
  hmcState = sampler.init(np.array([x, y]));
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
  stepCounter = 0;
  acceptedCount = 0;
  divergentCount = 0;

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

/**
 * Perform one HMC step.
 */
function performStep(): void {
  const key = random.key(stepCounter);
  const [newState, info] = sampler.step(key, hmcState);

  // Read info values before disposal
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;
  const isDivergent = info.isDivergent.ref.js() as boolean;
  const energy = info.energy.ref.js() as number;

  // Get current position for previous tracking
  const oldPos = hmcState.position.ref.js() as number[];
  prevX = oldPos[0];
  prevY = oldPos[1];

  // Dispose old state and info
  disposeInfo(info);

  // Update state
  hmcState = newState;
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

/**
 * Update statistics display.
 */
function updateStats(): void {
  statSamples.textContent = String(samples.length);

  if (samples.length > 0) {
    const rate = (acceptedCount / samples.length) * 100;
    statAcceptance.textContent = `${rate.toFixed(1)}%`;
    statDivergent.textContent = String(divergentCount);

    const meanX = samples.reduce((sum, s) => sum + s.x, 0) / samples.length;
    const meanY = samples.reduce((sum, s) => sum + s.y, 0) / samples.length;
    statMeanX.textContent = meanX.toFixed(3);
    statMeanY.textContent = meanY.toFixed(3);
  } else {
    statAcceptance.textContent = '-';
    statDivergent.textContent = '0';
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
  energy: number;
} | null): void {
  if (info === null) {
    infoAcceptProb.textContent = '-';
    infoEnergy.textContent = '-';
    infoStatus.textContent = '-';
    return;
  }

  infoAcceptProb.textContent = info.acceptanceProb.toFixed(3);
  infoEnergy.textContent = info.energy.toFixed(2);

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
 * Render the current visualization state.
 */
function render(): void {
  const posArray = hmcState.position.ref.js() as number[];
  const x = posArray[0]!;
  const y = posArray[1]!;
  renderer.render(contours, samples, x, y, prevX, prevY);
}

/**
 * Animation loop tick.
 */
function tick(): void {
  if (!isPlaying) return;

  performStep();

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
  loadingEl.querySelector('.loading-text')!.textContent = 'Initializing JAX-JS...';

  // Small delay to let UI render
  await new Promise((r) => setTimeout(r, 100));

  try {
    // Initialize distribution
    currentDistribution = distributions.gaussian!();

    // Create renderer
    renderer = new CanvasRenderer(canvas, currentDistribution.bounds);

    // Build sampler and state (this triggers JAX-JS warmup)
    loadingEl.querySelector('.loading-text')!.textContent = 'Building HMC sampler...';
    await new Promise((r) => setTimeout(r, 50));

    buildSampler();
    initializeState();

    // Compute contours
    loadingEl.querySelector('.loading-text')!.textContent = 'Computing contours...';
    await new Promise((r) => setTimeout(r, 50));
    computeContours();

    // Initial render
    render();

    // Setup UI
    setupEventListeners();

    // Hide loading
    loadingEl.style.display = 'none';
  } catch (error) {
    loadingEl.querySelector('.loading-text')!.textContent =
      `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    console.error('Initialization error:', error);
  }
}

// Start
init();
