// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

import { numpy as np, random, init as jaxInit, defaultDevice, type Array as JaxArray } from '@jax-js/jax';
import { HMC, RWM, type HMCInfo, type HMCState, type RWMInfo, type RWMState } from '../../../src';
import { computeDensityGrid, computeContourLevels, extractContours, type ContourLine, type Bounds } from '../contour';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Sample = { x: number; y: number; accepted: boolean };

type DataSet = {
  y: number[];
  t: number[];
  treated: number[];
  control: number[];
  seed: number;
};

type VizConfig = {
  effectSize: number;
  outlierRate: number;
  sampleSize: number;
  priorStrength: number;
  rwmStepSize: number;
  hmcStepSize: number;
  hmcSteps: number;
};

type VizState = {
  stepIndex: number;
  config: VizConfig;
  dataset: { seed: number; sampleSize: number };
  rwm: { steps: number; accepted: number; acceptanceRate: number; position: [number, number] };
  hmc: { steps: number; accepted: number; acceptanceRate: number; position: [number, number] };
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const canvas = document.getElementById('viz-canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const heatmapCanvas = document.createElement('canvas');
const heatmapCtx = heatmapCanvas.getContext('2d')!;
const loadingEl = document.getElementById('loading') as HTMLDivElement;
const legendEl = document.getElementById('legend') as HTMLDivElement;
const stepCounterEl = document.getElementById('step-counter') as HTMLDivElement;
const stepTitleEl = document.getElementById('step-title') as HTMLHeadingElement;
const stepDescEl = document.getElementById('step-desc') as HTMLParagraphElement;
const stepperEl = document.getElementById('stepper') as HTMLDivElement;
const prevStepBtn = document.getElementById('prev-step') as HTMLButtonElement;
const nextStepBtn = document.getElementById('next-step') as HTMLButtonElement;

const effectSizeSlider = document.getElementById('effect-size') as HTMLInputElement;
const effectSizeValue = document.getElementById('effect-size-value') as HTMLSpanElement;
const outlierRateSlider = document.getElementById('outlier-rate') as HTMLInputElement;
const outlierRateValue = document.getElementById('outlier-rate-value') as HTMLSpanElement;
const sampleSizeSlider = document.getElementById('sample-size') as HTMLInputElement;
const sampleSizeValue = document.getElementById('sample-size-value') as HTMLSpanElement;
const resampleButton = document.getElementById('resample-data') as HTMLButtonElement;

const priorStrengthSlider = document.getElementById('prior-strength') as HTMLInputElement;
const priorStrengthValue = document.getElementById('prior-strength-value') as HTMLSpanElement;

const rwmStepSlider = document.getElementById('rwm-step') as HTMLInputElement;
const rwmStepValue = document.getElementById('rwm-step-value') as HTMLSpanElement;
const hmcStepSlider = document.getElementById('hmc-step') as HTMLInputElement;
const hmcStepValue = document.getElementById('hmc-step-value') as HTMLSpanElement;
const hmcStepsSlider = document.getElementById('hmc-steps') as HTMLInputElement;
const hmcStepsValue = document.getElementById('hmc-steps-value') as HTMLSpanElement;
const playPauseBtn = document.getElementById('play-pause') as HTMLButtonElement;
const singleStepBtn = document.getElementById('single-step') as HTMLButtonElement;

// ---------------------------------------------------------------------------
// Config + Story
// ---------------------------------------------------------------------------

const BASELINE_MEAN = 70;
const REGULAR_SIGMA = 10;
const OUTLIER_SIGMA = 10;
const OUTLIER_SHIFT = -10;
const BASELINE_SD = 15;
const EFFECT_SD = 8;

const STEP_TITLES = [
  'Problem Setup',
  'Bayesian Update',
  'Posterior Geometry',
  'MCMC Sampling',
  'Causal Contrast',
];

const STEP_DESCRIPTIONS = [
  'We observe exam scores after a training program. Most noise is mild, but some outliers drag scores down.',
  'Prior beliefs meet data. The posterior is proportional to prior x likelihood, but the normalization is intractable.',
  'The mixture noise warps the posterior into a non-Gaussian shape -- even with simple priors.',
  'Random-walk Metropolis and HMC sample the same target. One struggles; one navigates geometry.',
  'Posterior predictive differences between trained and untrained outcomes -- the effect we care about.',
];

const STEP_LABELS = [
  'Setup',
  'Update',
  'Geometry',
  'Sampling',
  'Contrast',
];

const config: VizConfig = {
  effectSize: parseFloat(effectSizeSlider.value),
  outlierRate: parseFloat(outlierRateSlider.value),
  sampleSize: parseInt(sampleSizeSlider.value, 10),
  priorStrength: parseFloat(priorStrengthSlider.value),
  rwmStepSize: parseFloat(rwmStepSlider.value),
  hmcStepSize: parseFloat(hmcStepSlider.value),
  hmcSteps: parseInt(hmcStepsSlider.value, 10),
};

let currentStep = 0;
let dataset: DataSet;
let dataYArray: JaxArray | null = null;
let dataTArray: JaxArray | null = null;
let paramBounds: Bounds;
let massMatrix: JaxArray | null = null;

let priorGrid: number[][] | null = null;
let likelihoodGrid: number[][] | null = null;
let posteriorGrid: number[][] | null = null;
let posteriorContours: ContourLine[] = [];

let rwmSampler: { init: (pos: JaxArray) => RWMState; step: (key: JaxArray, state: RWMState) => [RWMState, RWMInfo] };
let hmcSampler: { init: (pos: JaxArray) => HMCState; step: (key: JaxArray, state: HMCState) => [HMCState, HMCInfo] };
let rwmState: RWMState;
let hmcState: HMCState;
let rwmSamples: Sample[] = [];
let hmcSamples: Sample[] = [];
let rwmAccepted = 0;
let hmcAccepted = 0;
let rwmStepCounter = 0;
let hmcStepCounter = 0;
let baseSeed = Date.now();
let isRunning = false;
let animationId: number | null = null;


// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function resizeCanvas(): void {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth * ratio;
  const height = canvas.clientHeight * ratio;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSample(rng: () => number, mean = 0, sd = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const mag = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + sd * mag;
}

function histogram(values: number[], bins: number, minVal?: number, maxVal?: number): { bins: number[]; edges: number[] } {
  if (values.length === 0) {
    return { bins: new Array(bins).fill(0), edges: new Array(bins + 1).fill(0) };
  }
  const min = minVal ?? Math.min(...values);
  const max = maxVal ?? Math.max(...values);
  const span = max - min || 1;
  const edges = new Array(bins + 1).fill(0).map((_, i) => min + (i / bins) * span);
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = clamp(Math.floor(((v - min) / span) * bins), 0, bins - 1);
    counts[idx] += 1;
  }
  return { bins: counts, edges };
}

function smoothSeries(values: number[], window = 3): number[] {
  if (values.length <= window) return values;
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = -window; j <= window; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < values.length) {
        sum += values[idx]!;
        count += 1;
      }
    }
    out.push(sum / count);
  }
  return out;
}

function computeParamBounds(): Bounds {
  const controlCount = dataset.control.length;
  const meanControl = controlCount
    ? dataset.control.reduce((a, b) => a + b, 0) / controlCount
    : dataset.y.reduce((a, b) => a + b, 0) / dataset.y.length;
  const baseRange = 25;
  return {
    xMin: meanControl - baseRange,
    xMax: meanControl + baseRange,
    yMin: -15,
    yMax: 15,
  };
}

function drawHeatmap(grid: number[][], bounds: Bounds, rect: { x: number; y: number; w: number; h: number }): void {
  const rows = grid.length;
  const cols = grid[0].length;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (const row of grid) {
    for (const val of row) {
      if (isFinite(val)) {
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
    }
  }
  if (!isFinite(minVal) || !isFinite(maxVal)) return;
  const range = maxVal - minVal || 1;

  if (heatmapCanvas.width !== cols || heatmapCanvas.height !== rows) {
    heatmapCanvas.width = cols;
    heatmapCanvas.height = rows;
  }
  const imageData = heatmapCtx.createImageData(cols, rows);
  const data = imageData.data;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const idx = ((rows - 1 - j) * cols + i) * 4;
      const t = clamp((grid[j][i] - minVal) / range, 0, 1);
      const color = [15, 42, 43];
      const color2 = [239, 245, 244];
      const r = Math.round(color[0] + (color2[0] - color[0]) * t);
      const g = Math.round(color[1] + (color2[1] - color[1]) * t);
      const b = Math.round(color[2] + (color2[2] - color[2]) * t);
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }
  heatmapCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(heatmapCanvas, rect.x, rect.y, rect.w, rect.h);

  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 0.6;
  const levels = computeContourLevels(grid, 6);
  const xs = linspace(bounds.xMin, bounds.xMax, cols);
  const ys = linspace(bounds.yMin, bounds.yMax, rows);
  const contours = extractContours(grid, xs, ys, levels);
  for (const line of contours) {
    ctx.beginPath();
    line.points.forEach(([x, y], idx) => {
      const px = rect.x + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * rect.w;
      const py = rect.y + rect.h - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * rect.h;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
}

function drawContours(lines: ContourLine[], bounds: Bounds, rect: { x: number; y: number; w: number; h: number }, stroke: string): void {
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1;
  for (const line of lines) {
    ctx.beginPath();
    line.points.forEach(([x, y], idx) => {
      const px = rect.x + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * rect.w;
      const py = rect.y + rect.h - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * rect.h;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
}

function drawAxis(rect: { x: number; y: number; w: number; h: number }, labelX: string, labelY: string): void {
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '11px "Alegreya Sans", sans-serif';
  ctx.fillText(labelX, rect.x + rect.w - 40, rect.y + rect.h + 18);
  ctx.save();
  ctx.translate(rect.x - 18, rect.y + 12);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(labelY, 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Data + Model
// ---------------------------------------------------------------------------

function generateData(seed: number): DataSet {
  const rng = mulberry32(seed);
  const y: number[] = [];
  const t: number[] = [];
  const treated: number[] = [];
  const control: number[] = [];
  for (let i = 0; i < config.sampleSize; i++) {
    const ti = rng() < 0.5 ? 1 : 0;
    const mu = BASELINE_MEAN + config.effectSize * ti;
    const isOutlier = rng() < config.outlierRate;
    const noise = isOutlier ? normalSample(rng, OUTLIER_SHIFT, OUTLIER_SIGMA) : normalSample(rng, 0, REGULAR_SIGMA);
    const yi = mu + noise;
    y.push(yi);
    t.push(ti);
    if (ti === 1) treated.push(yi);
    else control.push(yi);
  }
  return { y, t, treated, control, seed };
}

function refreshData(seed?: number): void {
  const nextSeed = seed ?? Date.now();
  dataset = generateData(nextSeed);

  dataYArray?.dispose();
  dataTArray?.dispose();

  dataYArray = np.array(dataset.y);
  dataTArray = np.array(dataset.t);
  paramBounds = computeParamBounds();

  recomputeGrids();
  resetSamplers(true);
}

function logPriorJS(b: number, e: number): number {
  const sdBase = BASELINE_SD * config.priorStrength;
  const sdEffect = EFFECT_SD * config.priorStrength;
  return -0.5 * ((b * b) / (sdBase * sdBase) + (e * e) / (sdEffect * sdEffect));
}

function logLikelihoodJS(b: number, e: number): number {
  const logNorm = -Math.log(REGULAR_SIGMA * Math.sqrt(2 * Math.PI));
  const logNormOut = -Math.log(OUTLIER_SIGMA * Math.sqrt(2 * Math.PI));
  const w1 = 1 - config.outlierRate;
  const w2 = config.outlierRate;
  let sum = 0;
  for (let i = 0; i < dataset.y.length; i++) {
    const mu = b + e * dataset.t[i]!;
    const y = dataset.y[i]!;
    const diff1 = y - mu;
    const logp1 = logNorm - 0.5 * (diff1 * diff1) / (REGULAR_SIGMA * REGULAR_SIGMA);
    if (w2 <= 0) {
      sum += logp1;
      continue;
    }
    const diff2 = y - (mu + OUTLIER_SHIFT);
    const logp2 = logNormOut - 0.5 * (diff2 * diff2) / (OUTLIER_SIGMA * OUTLIER_SIGMA);
    if (w1 <= 0) {
      sum += logp2;
      continue;
    }
    const logp1w = logp1 + Math.log(w1);
    const logp2w = logp2 + Math.log(w2);
    const maxLog = Math.max(logp1w, logp2w);
    sum += maxLog + Math.log(Math.exp(logp1w - maxLog) + Math.exp(logp2w - maxLog));
  }
  return sum;
}

function logPosteriorJS(b: number, e: number): number {
  return logPriorJS(b, e) + logLikelihoodJS(b, e);
}

function logdensity(position: JaxArray): JaxArray {
  if (!dataYArray || !dataTArray) {
    throw new Error('Data arrays not initialized');
  }
  const b = position.ref.slice([0, 1]).reshape([]);
  const e = position.slice([1, 2]).reshape([]);

  const mu = dataTArray.ref.mul(e.ref).add(b.ref);
  const diff1 = dataYArray.ref.sub(mu.ref);
  const logNorm = -Math.log(REGULAR_SIGMA * Math.sqrt(2 * Math.PI));
  const logNormOut = -Math.log(OUTLIER_SIGMA * Math.sqrt(2 * Math.PI));

  const logp1 = np.square(diff1.ref).mul(-0.5 / (REGULAR_SIGMA * REGULAR_SIGMA)).add(logNorm);

  const muOut = mu.sub(OUTLIER_SHIFT);
  const diff2 = dataYArray.ref.sub(muOut);
  const logp2 = np.square(diff2).mul(-0.5 / (OUTLIER_SIGMA * OUTLIER_SIGMA)).add(logNormOut);

  const w1 = Math.max(1 - config.outlierRate, 0);
  const w2 = Math.max(config.outlierRate, 0);

  let logLike: JaxArray;
  if (w2 <= 0) {
    logLike = logp1;
  } else if (w1 <= 0) {
    logLike = logp2;
  } else {
    const logp1w = logp1.add(Math.log(w1));
    const logp2w = logp2.add(Math.log(w2));
    const maxLog = np.maximum(logp1w.ref, logp2w.ref);
    const sumExp = np.exp(logp1w.sub(maxLog.ref)).add(np.exp(logp2w.sub(maxLog.ref)));
    logLike = maxLog.add(np.log(sumExp));
  }

  const logLikeSum = logLike.sum();
  const sdBase = BASELINE_SD * config.priorStrength;
  const sdEffect = EFFECT_SD * config.priorStrength;
  const prior = b.ref.mul(b).mul(-0.5 / (sdBase * sdBase)).add(e.ref.mul(e).mul(-0.5 / (sdEffect * sdEffect)));

  return logLikeSum.add(prior);
}

function recomputeGrids(): void {
  const resolution = 60;
  const { grid: prior } = computeDensityGrid(logPriorJS, paramBounds, resolution);
  const { grid: like } = computeDensityGrid(logLikelihoodJS, paramBounds, resolution);
  const { grid: post } = computeDensityGrid(logPosteriorJS, paramBounds, resolution);
  priorGrid = prior;
  likelihoodGrid = like;
  posteriorGrid = post;

  const levels = computeContourLevels(post, 10);
  posteriorContours = extractContours(post, linspace(paramBounds.xMin, paramBounds.xMax, resolution), linspace(paramBounds.yMin, paramBounds.yMax, resolution), levels);
}

function linspace(min: number, max: number, n: number): number[] {
  const out: number[] = [];
  const step = (max - min) / (n - 1);
  for (let i = 0; i < n; i++) out.push(min + step * i);
  return out;
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

function buildSamplers(): void {
  rwmSampler = RWM(logdensity).stepSize(config.rwmStepSize).build();
  hmcSampler = HMC(logdensity)
    .stepSize(config.hmcStepSize)
    .numIntegrationSteps(config.hmcSteps)
    .inverseMassMatrix(massMatrix ? massMatrix.ref : np.array([1.0, 1.0]))
    .build();
}

function resetSamplers(rebuild: boolean): void {
  if (rebuild) buildSamplers();

  const initB = clamp(BASELINE_MEAN, paramBounds.xMin + 2, paramBounds.xMax - 2);
  const initE = clamp(config.effectSize, paramBounds.yMin + 1, paramBounds.yMax - 1);
  rwmState = rwmSampler.init(np.array([initB, initE]));
  hmcState = hmcSampler.init(np.array([initB, initE]));

  rwmSamples = [];
  hmcSamples = [];
  rwmAccepted = 0;
  hmcAccepted = 0;
  rwmStepCounter = 0;
  hmcStepCounter = 0;
}

function disposeRWMInfo(info: RWMInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.proposedPosition.dispose();
}

function disposeHMCInfo(info: HMCInfo): void {
  info.acceptanceProb.dispose();
  info.isAccepted.dispose();
  info.momentum.dispose();
  info.isDivergent.dispose();
  info.energy.dispose();
}

function performRwmStep(): void {
  const key = random.key(baseSeed + rwmStepCounter + 1000);
  const [newState, info] = rwmSampler.step(key, rwmState);
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;

  const posArray = newState.position.ref.js() as number[];
  rwmSamples.push({ x: posArray[0]!, y: posArray[1]!, accepted: isAccepted });
  if (isAccepted) rwmAccepted++;

  disposeRWMInfo(info);
  rwmState = newState;
  rwmStepCounter++;

  if (rwmSamples.length > 600) rwmSamples = rwmSamples.slice(-600);
}

function performHmcStep(): void {
  const key = random.key(baseSeed + hmcStepCounter);
  const [newState, info] = hmcSampler.step(key, hmcState);
  const acceptanceProb = info.acceptanceProb.ref.js() as number;
  const isAccepted = info.isAccepted.ref.js() as boolean;

  const posArray = newState.position.ref.js() as number[];
  hmcSamples.push({ x: posArray[0]!, y: posArray[1]!, accepted: isAccepted });
  if (isAccepted) hmcAccepted++;

  disposeHMCInfo(info);
  hmcState = newState;
  hmcStepCounter++;

  if (hmcSamples.length > 600) hmcSamples = hmcSamples.slice(-600);
}

function performSamplingStep(): void {
  performRwmStep();
  performHmcStep();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render(): void {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = '#0f2a2b';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  switch (currentStep) {
    case 0:
      drawHistogramStep();
      break;
    case 1:
      drawBayesUpdateStep();
      break;
    case 2:
      drawPosteriorGeometryStep();
      break;
    case 3:
      drawSamplingStep();
      break;
    case 4:
      drawContrastStep();
      break;
    default:
      break;
  }
}

function drawHistogramStep(): void {
  const padding = 30;
  const rect = { x: padding, y: padding, w: canvas.clientWidth - padding * 2, h: canvas.clientHeight - padding * 2 };
  const bins = 24;

  const all = dataset.y;
  const minVal = Math.min(...all) - 3;
  const maxVal = Math.max(...all) + 3;
  const histControl = histogram(dataset.control, bins, minVal, maxVal);
  const histTreat = histogram(dataset.treated, bins, minVal, maxVal);
  const maxCount = Math.max(...histControl.bins, ...histTreat.bins) || 1;

  ctx.fillStyle = '#0f2a2b';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  drawAxis(rect, 'Score', 'Count');

  for (let i = 0; i < bins; i++) {
    const x0 = rect.x + (i / bins) * rect.w;
    const barW = rect.w / bins;
    const hControl = (histControl.bins[i]! / maxCount) * rect.h;
    const hTreat = (histTreat.bins[i]! / maxCount) * rect.h;

    ctx.fillStyle = 'rgba(31, 138, 112, 0.55)';
    ctx.fillRect(x0, rect.y + rect.h - hControl, barW - 2, hControl);

    ctx.fillStyle = 'rgba(244, 162, 97, 0.55)';
    ctx.fillRect(x0 + 2, rect.y + rect.h - hTreat, barW - 2, hTreat);
  }

  legendEl.innerHTML = `
    <span><span class="legend-dot" style="background:#1f8a70"></span>Control</span>
    <span><span class="legend-dot" style="background:#f4a261"></span>Trained</span>
  `;
}

function drawBayesUpdateStep(): void {
  if (!priorGrid || !likelihoodGrid || !posteriorGrid) return;

  const padding = 20;
  const totalW = canvas.clientWidth - padding * 2;
  const panelW = (totalW - 20 * 2) / 3;
  const panelH = canvas.clientHeight - padding * 2;

  const panels = [
    { title: 'Prior', grid: priorGrid },
    { title: 'Likelihood', grid: likelihoodGrid },
    { title: 'Posterior', grid: posteriorGrid },
  ];

  ctx.fillStyle = '#0f2a2b';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  panels.forEach((panel, idx) => {
    const x = padding + idx * (panelW + 20);
    const y = padding;
    drawHeatmap(panel.grid, paramBounds, { x, y, w: panelW, h: panelH - 30 });

    ctx.fillStyle = 'rgba(239,245,244,0.9)';
    ctx.font = '12px "Alegreya Sans", sans-serif';
    ctx.fillText(panel.title, x + 8, y + panelH - 8);
  });

  legendEl.innerHTML = `
    <span><span class="legend-dot" style="background:#eff5f4"></span>High density</span>
    <span><span class="legend-dot" style="background:#0f2a2b"></span>Low density</span>
  `;
}

function drawPosteriorGeometryStep(): void {
  if (!posteriorGrid) return;
  const padding = 30;
  const rect = { x: padding, y: padding, w: canvas.clientWidth - padding * 2, h: canvas.clientHeight - padding * 2 };

  drawHeatmap(posteriorGrid, paramBounds, rect);
  drawContours(posteriorContours, paramBounds, rect, 'rgba(244, 162, 97, 0.75)');

  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '12px "Alegreya Sans", sans-serif';
  ctx.fillText('baseline', rect.x + rect.w - 50, rect.y + rect.h + 16);
  ctx.save();
  ctx.translate(rect.x - 14, rect.y + 14);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('effect', 0, 0);
  ctx.restore();

  legendEl.innerHTML = `
    <span><span class="legend-dot" style="background:#f4a261"></span>Posterior contours</span>
  `;
}

function drawSamplingStep(): void {
  if (!posteriorGrid) return;
  const padding = 20;
  const topH = canvas.clientHeight * 0.72;
  const rect = { x: padding, y: padding, w: canvas.clientWidth - padding * 2, h: topH - padding };

  drawHeatmap(posteriorGrid, paramBounds, rect);
  drawContours(posteriorContours, paramBounds, rect, 'rgba(255,255,255,0.25)');

  drawSamples(hmcSamples, rect, '#1f8a70');
  drawSamples(rwmSamples, rect, '#f4a261');

  const traceRect = {
    x: padding,
    y: topH + 10,
    w: canvas.clientWidth - padding * 2,
    h: canvas.clientHeight - topH - 30,
  };
  drawTrace(traceRect);

  legendEl.innerHTML = `
    <span><span class="legend-dot" style="background:#1f8a70"></span>HMC samples</span>
    <span><span class="legend-dot" style="background:#f4a261"></span>RWM samples</span>
  `;
}

function drawSamples(samples: Sample[], rect: { x: number; y: number; w: number; h: number }, color: string): void {
  for (const s of samples) {
    const px = rect.x + ((s.x - paramBounds.xMin) / (paramBounds.xMax - paramBounds.xMin)) * rect.w;
    const py = rect.y + rect.h - ((s.y - paramBounds.yMin) / (paramBounds.yMax - paramBounds.yMin)) * rect.h;
    ctx.fillStyle = s.accepted ? color : `${color}55`;
    ctx.beginPath();
    ctx.arc(px, py, s.accepted ? 2.2 : 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawTrace(rect: { x: number; y: number; w: number; h: number }): void {
  const maxLen = 140;
  const rwmTrace = rwmSamples.slice(-maxLen).map((s) => s.y);
  const hmcTrace = hmcSamples.slice(-maxLen).map((s) => s.y);
  const minVal = Math.min(paramBounds.yMin, ...rwmTrace, ...hmcTrace);
  const maxVal = Math.max(paramBounds.yMax, ...rwmTrace, ...hmcTrace);

  ctx.fillStyle = 'rgba(15,42,43,0.5)';
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

  const drawLine = (trace: number[], color: string) => {
    if (trace.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    trace.forEach((val, idx) => {
      const x = rect.x + (idx / (maxLen - 1)) * rect.w;
      const y = rect.y + rect.h - ((val - minVal) / (maxVal - minVal || 1)) * rect.h;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  drawLine(hmcTrace, '#1f8a70');
  drawLine(rwmTrace, '#f4a261');

  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '11px "Alegreya Sans", sans-serif';
  ctx.fillText('effect trace', rect.x + 6, rect.y + 12);
}

function drawContrastStep(): void {
  const padding = 30;
  const rect = { x: padding, y: padding, w: canvas.clientWidth - padding * 2, h: canvas.clientHeight - padding * 2 };

  const diffs = computePredictiveDiffs();
  const hist = histogram(diffs, 40);
  const maxCount = Math.max(...hist.bins) || 1;
  const smooth = smoothSeries(hist.bins, 2);

  ctx.fillStyle = '#0f2a2b';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

  const zeroX = rect.x + ((0 - hist.edges[0]) / (hist.edges[hist.edges.length - 1]! - hist.edges[0] || 1)) * rect.w;
  ctx.strokeStyle = 'rgba(244, 162, 97, 0.8)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(zeroX, rect.y);
  ctx.lineTo(zeroX, rect.y + rect.h);
  ctx.stroke();

  ctx.strokeStyle = '#1f8a70';
  ctx.lineWidth = 2;
  ctx.beginPath();
  smooth.forEach((count, idx) => {
    const x = rect.x + (idx / (smooth.length - 1)) * rect.w;
    const y = rect.y + rect.h - (count / maxCount) * rect.h;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '12px "Alegreya Sans", sans-serif';
  ctx.fillText('difference (trained - control)', rect.x + rect.w - 180, rect.y + rect.h + 18);

  legendEl.innerHTML = `
    <span><span class="legend-dot" style="background:#1f8a70"></span>Posterior predictive contrast</span>
    <span><span class="legend-dot" style="background:#f4a261"></span>Zero effect</span>
  `;
}

function computePredictiveDiffs(): number[] {
  const rng = mulberry32(dataset.seed + 1234);
  const samples = hmcSamples.length > 80 ? hmcSamples.slice(-80) : hmcSamples;
  const diffs: number[] = [];
  for (const s of samples) {
    for (let i = 0; i < 8; i++) {
      const noiseT = sampleNoise(rng);
      const noiseC = sampleNoise(rng);
      diffs.push(s.y + noiseT - noiseC);
    }
  }
  if (diffs.length === 0) {
    for (let i = 0; i < 200; i++) {
      diffs.push(normalSample(rng, config.effectSize, 6));
    }
  }
  return diffs;
}

function sampleNoise(rng: () => number): number {
  const isOutlier = rng() < config.outlierRate;
  return isOutlier ? normalSample(rng, OUTLIER_SHIFT, OUTLIER_SIGMA) : normalSample(rng, 0, REGULAR_SIGMA);
}

// ---------------------------------------------------------------------------
// UI + State
// ---------------------------------------------------------------------------

function updateStepUI(): void {
  stepCounterEl.textContent = `Step ${currentStep + 1} of 5`;
  stepTitleEl.textContent = STEP_TITLES[currentStep] ?? '';
  stepDescEl.textContent = STEP_DESCRIPTIONS[currentStep] ?? '';

  prevStepBtn.disabled = currentStep === 0;
  nextStepBtn.disabled = currentStep === STEP_TITLES.length - 1;

  const pills = Array.from(stepperEl.querySelectorAll('.step-pill')) as HTMLDivElement[];
  pills.forEach((pill, idx) => {
    if (idx === currentStep) pill.classList.add('active');
    else pill.classList.remove('active');
  });

  const panels = document.querySelectorAll('.control-panel');
  panels.forEach((panel) => {
    const steps = panel.getAttribute('data-steps');
    if (!steps) return;
    const indices = steps.split(',').map((s) => parseInt(s.trim(), 10));
    panel.classList.toggle('hidden', !indices.includes(currentStep));
  });

  if (currentStep !== 3 && isRunning) {
    toggleRunning(false);
  }

  render();
}

function updateValueLabels(): void {
  effectSizeValue.textContent = formatNumber(config.effectSize, 1);
  outlierRateValue.textContent = formatNumber(config.outlierRate, 2);
  sampleSizeValue.textContent = String(config.sampleSize);
  priorStrengthValue.textContent = formatNumber(config.priorStrength, 2);
  rwmStepValue.textContent = formatNumber(config.rwmStepSize, 2);
  hmcStepValue.textContent = formatNumber(config.hmcStepSize, 2);
  hmcStepsValue.textContent = String(config.hmcSteps);
}

function toggleRunning(force?: boolean): void {
  const next = force ?? !isRunning;
  isRunning = next;
  playPauseBtn.textContent = isRunning ? 'Pause' : 'Play';
  if (isRunning) {
    animate();
  } else if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

function animate(): void {
  if (!isRunning) return;
  for (let i = 0; i < 2; i++) {
    performSamplingStep();
  }
  render();
  animationId = requestAnimationFrame(animate);
}

function setStep(index: number): void {
  currentStep = clamp(index, 0, STEP_TITLES.length - 1);
  updateStepUI();
}

function createStepper(): void {
  STEP_LABELS.forEach((label, idx) => {
    const pill = document.createElement('div');
    pill.className = 'step-pill';
    pill.textContent = `${idx + 1}. ${label}`;
    stepperEl.appendChild(pill);
  });
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

prevStepBtn.addEventListener('click', () => setStep(currentStep - 1));
nextStepBtn.addEventListener('click', () => setStep(currentStep + 1));

resampleButton.addEventListener('click', () => {
  refreshData();
  render();
});

effectSizeSlider.addEventListener('input', () => {
  config.effectSize = parseFloat(effectSizeSlider.value);
  updateValueLabels();
});

outlierRateSlider.addEventListener('input', () => {
  config.outlierRate = parseFloat(outlierRateSlider.value);
  updateValueLabels();
});

sampleSizeSlider.addEventListener('input', () => {
  config.sampleSize = parseInt(sampleSizeSlider.value, 10);
  updateValueLabels();
});

priorStrengthSlider.addEventListener('input', () => {
  config.priorStrength = parseFloat(priorStrengthSlider.value);
  updateValueLabels();
  recomputeGrids();
  resetSamplers(true);
  render();
});

rwmStepSlider.addEventListener('input', () => {
  config.rwmStepSize = parseFloat(rwmStepSlider.value);
  updateValueLabels();
  rwmSampler = RWM(logdensity).stepSize(config.rwmStepSize).build();
});

hmcStepSlider.addEventListener('input', () => {
  config.hmcStepSize = parseFloat(hmcStepSlider.value);
  updateValueLabels();
  hmcSampler = HMC(logdensity)
    .stepSize(config.hmcStepSize)
    .numIntegrationSteps(config.hmcSteps)
    .inverseMassMatrix(massMatrix ? massMatrix.ref : np.array([1.0, 1.0]))
    .build();
});

hmcStepsSlider.addEventListener('input', () => {
  config.hmcSteps = parseInt(hmcStepsSlider.value, 10);
  updateValueLabels();
  hmcSampler = HMC(logdensity)
    .stepSize(config.hmcStepSize)
    .numIntegrationSteps(config.hmcSteps)
    .inverseMassMatrix(massMatrix ? massMatrix.ref : np.array([1.0, 1.0]))
    .build();
});

playPauseBtn.addEventListener('click', () => toggleRunning());

singleStepBtn.addEventListener('click', () => {
  performSamplingStep();
  render();
});

window.addEventListener('resize', render);

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

function buildState(): VizState {
  const rwmPos = rwmState.position.ref.js() as number[];
  const hmcPos = hmcState.position.ref.js() as number[];
  return {
    stepIndex: currentStep,
    config: { ...config },
    dataset: { seed: dataset.seed, sampleSize: dataset.y.length },
    rwm: {
      steps: rwmStepCounter,
      accepted: rwmAccepted,
      acceptanceRate: rwmStepCounter ? rwmAccepted / rwmStepCounter : 0,
      position: [rwmPos[0]!, rwmPos[1]!],
    },
    hmc: {
      steps: hmcStepCounter,
      accepted: hmcAccepted,
      acceptanceRate: hmcStepCounter ? hmcAccepted / hmcStepCounter : 0,
      position: [hmcPos[0]!, hmcPos[1]!],
    },
  };
}

(window as unknown as { __vizDebug: { getState: () => VizState; step: () => VizState; reset: () => VizState; setConfig: (config: Partial<VizConfig> & { stepIndex?: number }) => VizState; } }).__vizDebug = {
  getState: () => buildState(),
  step: () => {
    performSamplingStep();
    render();
    return buildState();
  },
  reset: () => {
    resetSamplers(true);
    render();
    return buildState();
  },
  setConfig: (next) => {
    if (next.stepIndex !== undefined) {
      setStep(next.stepIndex);
    }
    if (next.effectSize !== undefined) {
      config.effectSize = next.effectSize;
      effectSizeSlider.value = String(next.effectSize);
    }
    if (next.outlierRate !== undefined) {
      config.outlierRate = next.outlierRate;
      outlierRateSlider.value = String(next.outlierRate);
    }
    if (next.sampleSize !== undefined) {
      config.sampleSize = next.sampleSize;
      sampleSizeSlider.value = String(next.sampleSize);
    }
    if (next.priorStrength !== undefined) {
      config.priorStrength = next.priorStrength;
      priorStrengthSlider.value = String(next.priorStrength);
    }
    if (next.rwmStepSize !== undefined) {
      config.rwmStepSize = next.rwmStepSize;
      rwmStepSlider.value = String(next.rwmStepSize);
    }
    if (next.hmcStepSize !== undefined) {
      config.hmcStepSize = next.hmcStepSize;
      hmcStepSlider.value = String(next.hmcStepSize);
    }
    if (next.hmcSteps !== undefined) {
      config.hmcSteps = next.hmcSteps;
      hmcStepsSlider.value = String(next.hmcSteps);
    }

    updateValueLabels();
    refreshData(dataset.seed);
    rwmSampler = RWM(logdensity).stepSize(config.rwmStepSize).build();
    hmcSampler = HMC(logdensity)
      .stepSize(config.hmcStepSize)
      .numIntegrationSteps(config.hmcSteps)
      .inverseMassMatrix(massMatrix ? massMatrix.ref : np.array([1.0, 1.0]))
      .build();
    render();
    return buildState();
  },
};

console.log('[BAYES-MCMC] Debug API exposed to window.__vizDebug');

async function debugPollLoop(): Promise<void> {
  if (import.meta.env.PROD) return;

  while (true) {
    try {
      const res = await fetch('/__debug/poll');
      const cmd = await res.json() as { id: string; type: string; payload?: unknown } | null;

      if (cmd?.type) {
        const api = (window as unknown as { __vizDebug: { getState: () => VizState; step: () => VizState; reset: () => VizState; setConfig: (config: Partial<VizConfig> & { stepIndex?: number }) => VizState; } }).__vizDebug;
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
            result = api.setConfig(cmd.payload as Partial<VizConfig> & { stepIndex?: number });
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

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init(): Promise<void> {
  updateValueLabels();
  createStepper();

  try {
    const availableDevices = await jaxInit();
    if (availableDevices.includes('webgpu')) {
      defaultDevice('webgpu');
    }

    massMatrix = np.array([1.0, 1.0]);
    refreshData(Date.now());
    updateStepUI();
  } catch (error) {
    console.error('[BAYES-MCMC] init error:', error);
  } finally {
    loadingEl.classList.add('hidden');
    render();
  }

  debugPollLoop();
}

init();
