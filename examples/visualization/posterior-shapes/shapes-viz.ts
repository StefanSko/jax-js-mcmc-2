/**
 * Posterior Shapes Visualization
 *
 * Pure JS visualization showing how data builds a likelihood, which shapes
 * a posterior that may no longer be closed-form.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type StepMode = 'likelihood' | 'posterior';

interface DataPoint {
  x: number;
  y: 0 | 1;
}

interface GridResult {
  xs: number[];
  prior: number[];
  likelihood: number[];
  posterior: number[];
  posteriorMode: number;
  posteriorMean: number;
  posteriorStd: number;
  posteriorPredictiveMean: number[];
  posteriorPredictiveStd: number[];
  decisionBoundary: number | null;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

const SQRT_2PI = Math.sqrt(2 * Math.PI);

function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * SQRT_2PI);
}

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function logSigmoid(x: number): number {
  if (x >= 0) {
    return -Math.log1p(Math.exp(-x));
  }
  return x - Math.log1p(Math.exp(x));
}

function log1mSigmoid(x: number): number {
  if (x >= 0) {
    return -x - Math.log1p(Math.exp(-x));
  }
  return -Math.log1p(Math.exp(x));
}

// ---------------------------------------------------------------------------
// Density functions
// ---------------------------------------------------------------------------

/** Standard normal prior N(0, 1) */
function prior(x: number): number {
  return gaussianPdf(x, 0, 1);
}

// ---------------------------------------------------------------------------
// Grid computation
// ---------------------------------------------------------------------------

const NUM_GRID_POINTS = 1000;
const X_MIN = -5;
const X_MAX = 5;
const LIKELIHOOD_SLOPE = 3.2;
const MAX_DATA_POINTS = 60;
const PREDICTIVE_BAND_SIGMA = 1.0;

const DEFAULT_DATA: DataPoint[] = [
  { x: -2.4, y: 0 },
  { x: -1.8, y: 0 },
  { x: -1.2, y: 0 },
  { x: -0.6, y: 0 },
  { x: 0.6, y: 1 },
  { x: 1.2, y: 1 },
  { x: 1.8, y: 1 },
  { x: 2.4, y: 1 },
];

const PRESETS: Record<string, DataPoint[]> = {
  separable: DEFAULT_DATA,
  overlap: [
    { x: -2.2, y: 0 },
    { x: -1.6, y: 0 },
    { x: -0.8, y: 0 },
    { x: -0.4, y: 1 },
    { x: 0.2, y: 0 },
    { x: 0.6, y: 1 },
    { x: 1.0, y: 0 },
    { x: 1.4, y: 1 },
    { x: 2.0, y: 1 },
  ],
  contradict: [
    { x: -1.4, y: 0 },
    { x: -1.4, y: 1 },
    { x: -0.4, y: 0 },
    { x: -0.4, y: 1 },
    { x: 0.6, y: 0 },
    { x: 0.6, y: 1 },
    { x: 1.4, y: 0 },
    { x: 1.4, y: 1 },
  ],
};

function computeGrid(data: DataPoint[]): GridResult {
  const dx = (X_MAX - X_MIN) / (NUM_GRID_POINTS - 1);
  const xs: number[] = [];
  const priorVals: number[] = [];
  const logLikelihoodVals: number[] = [];
  const likelihoodVals: number[] = [];
  const unnormalized: number[] = [];

  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    const x = X_MIN + i * dx;
    xs.push(x);

    const p = prior(x);
    priorVals.push(p);

    let logLik = 0;
    for (let j = 0; j < data.length; j++) {
      const point = data[j]!;
      const z = LIKELIHOOD_SLOPE * (point.x - x);
      logLik += point.y === 1 ? logSigmoid(z) : log1mSigmoid(z);
    }
    logLikelihoodVals.push(logLik);
  }

  // Normalize likelihood for plotting
  let maxLogLik = -Infinity;
  for (let i = 0; i < logLikelihoodVals.length; i++) {
    if (logLikelihoodVals[i]! > maxLogLik) maxLogLik = logLikelihoodVals[i]!;
  }
  if (!Number.isFinite(maxLogLik)) maxLogLik = 0;

  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    const lik = Math.exp(logLikelihoodVals[i]! - maxLogLik);
    likelihoodVals.push(lik);
    unnormalized.push(priorVals[i]! * lik);
  }

  // Normalize posterior using trapezoidal rule
  let integral = 0;
  for (let i = 0; i < NUM_GRID_POINTS - 1; i++) {
    integral += 0.5 * (unnormalized[i]! + unnormalized[i + 1]!) * dx;
  }

  const posteriorVals: number[] = [];
  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    posteriorVals.push(integral > 0 ? unnormalized[i]! / integral : 0);
  }

  // Find mode (argmax)
  let maxVal = -Infinity;
  let modeIdx = 0;
  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    if (posteriorVals[i]! > maxVal) {
      maxVal = posteriorVals[i]!;
      modeIdx = i;
    }
  }
  const posteriorMode = xs[modeIdx]!;

  // Posterior mean / std for quick stats
  let mean = 0;
  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    mean += xs[i]! * posteriorVals[i]! * dx;
  }
  let variance = 0;
  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    const diff = xs[i]! - mean;
    variance += diff * diff * posteriorVals[i]! * dx;
  }
  const posteriorStd = Math.sqrt(Math.max(0, variance));

  // Posterior predictive mean + std curve
  const predictiveStats = computePosteriorPredictiveStats(xs, posteriorVals, dx);
  const decisionBoundary = computeDecisionBoundary(xs, predictiveStats.mean);

  return {
    xs,
    prior: priorVals,
    likelihood: likelihoodVals,
    posterior: posteriorVals,
    posteriorMode,
    posteriorMean: mean,
    posteriorStd,
    posteriorPredictiveMean: predictiveStats.mean,
    posteriorPredictiveStd: predictiveStats.std,
    decisionBoundary,
  };
}

function computePosteriorPredictiveStats(
  xs: number[],
  posterior: number[],
  dx: number,
): { mean: number[]; std: number[] } {
  const mean: number[] = new Array(xs.length).fill(0);
  const std: number[] = new Array(xs.length).fill(0);

  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!;
    let sum = 0;
    let sumSq = 0;
    for (let j = 0; j < xs.length; j++) {
      const theta = xs[j]!;
      const p = sigmoid(LIKELIHOOD_SLOPE * (x - theta));
      const weight = posterior[j]! * dx;
      sum += weight * p;
      sumSq += weight * p * p;
    }
    const value = Math.min(1, Math.max(0, sum));
    const variance = Math.max(0, sumSq - value * value);
    mean[i] = value;
    std[i] = Math.sqrt(variance);
  }

  return { mean, std };
}

function computeDecisionBoundary(xs: number[], predictive: number[]): number | null {
  if (xs.length === 0 || xs.length !== predictive.length) return null;
  const target = 0.5;
  for (let i = 0; i < predictive.length - 1; i++) {
    const y0 = predictive[i]! - target;
    const y1 = predictive[i + 1]! - target;
    if (y0 === 0) return xs[i]!;
    if (y0 * y1 < 0) {
      const t = y0 / (y0 - y1);
      return xs[i]! + t * (xs[i + 1]! - xs[i]!);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

const COLORS = {
  prior: '#a78bfa',
  priorFill: 'rgba(167, 139, 250, 0.30)',
  likelihood: '#f472b6',
  likelihoodFill: 'rgba(244, 114, 182, 0.30)',
  posterior: '#4ade80',
  posteriorFill: 'rgba(74, 222, 128, 0.30)',
  predictive: '#60a5fa',
  predictiveFill: 'rgba(96, 165, 250, 0.25)',
  predictiveBand: 'rgba(96, 165, 250, 0.15)',
  dataYes: '#4ade80',
  dataNo: '#f87171',
  background: '#16213e',
  axis: '#334155',
  tick: '#475569',
  label: '#94a3b8',
  modeLine: '#facc15',
};

interface CanvasInfo {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

function getCanvasInfo(id: string): CanvasInfo {
  const canvas = document.getElementById(id) as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  return { canvas, ctx };
}

/**
 * Resize a canvas to match its CSS display size (for crisp rendering on HiDPI).
 */
function resizeCanvas(canvas: HTMLCanvasElement): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function drawCurve(
  info: CanvasInfo,
  xs: number[],
  ys: number[],
  strokeColor: string,
  fillColor: string,
  title: string,
  modeLine?: number,
): void {
  const { canvas, ctx } = info;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width;
  const H = canvas.height;

  const marginTop = 40 * dpr;
  const marginBottom = 36 * dpr;
  const marginLeft = 10 * dpr;
  const marginRight = 10 * dpr;

  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;

  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, W, H);

  let yMax = -Infinity;
  for (let i = 0; i < ys.length; i++) {
    if (ys[i]! > yMax) yMax = ys[i]!;
  }
  if (yMax <= 0) yMax = 1;
  yMax *= 1.1;

  const xMin = xs[0]!;
  const xMax = xs[xs.length - 1]!;

  const toPixelX = (x: number): number =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotW;
  const toPixelY = (y: number): number =>
    marginTop + plotH - (y / yMax) * plotH;

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop + plotH);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH);
  ctx.stroke();

  ctx.fillStyle = COLORS.tick;
  ctx.font = `${10 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let tickVal = Math.ceil(xMin); tickVal <= Math.floor(xMax); tickVal++) {
    const px = toPixelX(tickVal);
    ctx.beginPath();
    ctx.moveTo(px, marginTop + plotH);
    ctx.lineTo(px, marginTop + plotH + 4 * dpr);
    ctx.strokeStyle = COLORS.tick;
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();
    ctx.fillText(String(tickVal), px, marginTop + plotH + 6 * dpr);
  }

  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(0));
  for (let i = 0; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.lineTo(toPixelX(xs[xs.length - 1]!), toPixelY(0));
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(ys[0]!));
  for (let i = 1; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2 * dpr;
  ctx.stroke();

  if (modeLine !== undefined) {
    const mx = toPixelX(modeLine);
    ctx.save();
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.strokeStyle = COLORS.modeLine;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(mx, marginTop);
    ctx.lineTo(mx, marginTop + plotH);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = COLORS.modeLine;
    ctx.font = `${10 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`mode=${modeLine.toFixed(2)}`, mx, marginTop - 2 * dpr);
  }

  ctx.fillStyle = COLORS.label;
  ctx.font = `bold ${14 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(title, W / 2, 10 * dpr);
}

const DATA_STRIP_LAYOUT = {
  marginLeft: 28,
  marginRight: 16,
  marginTop: 16,
  marginBottom: 16,
  topRow: 0.25,
  bottomRow: 0.75,
};

function drawDataStrip(info: CanvasInfo, data: DataPoint[]): void {
  const { canvas, ctx } = info;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width;
  const H = canvas.height;

  const marginLeft = DATA_STRIP_LAYOUT.marginLeft * dpr;
  const marginRight = DATA_STRIP_LAYOUT.marginRight * dpr;
  const marginTop = DATA_STRIP_LAYOUT.marginTop * dpr;
  const marginBottom = DATA_STRIP_LAYOUT.marginBottom * dpr;

  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;

  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, W, H);

  const xMin = X_MIN;
  const xMax = X_MAX;

  const toPixelX = (x: number): number =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotW;

  const yYes = marginTop + plotH * DATA_STRIP_LAYOUT.topRow;
  const yNo = marginTop + plotH * DATA_STRIP_LAYOUT.bottomRow;

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop + plotH / 2);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH / 2);
  ctx.stroke();

  ctx.fillStyle = COLORS.label;
  ctx.font = `${10 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('y=1', marginLeft - 6 * dpr, yYes);
  ctx.fillText('y=0', marginLeft - 6 * dpr, yNo);

  for (let tickVal = Math.ceil(xMin); tickVal <= Math.floor(xMax); tickVal++) {
    const px = toPixelX(tickVal);
    ctx.beginPath();
    ctx.moveTo(px, marginTop + plotH / 2 - 4 * dpr);
    ctx.lineTo(px, marginTop + plotH / 2 + 4 * dpr);
    ctx.strokeStyle = COLORS.tick;
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();
  }

  const radius = 4 * dpr;
  for (let i = 0; i < data.length; i++) {
    const point = data[i]!;
    const px = toPixelX(point.x);
    const py = point.y === 1 ? yYes : yNo;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fillStyle = point.y === 1 ? COLORS.dataYes : COLORS.dataNo;
    ctx.fill();
  }
}

function jitterOffset(index: number, magnitude: number): number {
  const seed = (index * 9301 + 49297) % 233280;
  return ((seed / 233280) - 0.5) * magnitude;
}

function drawPredictive(
  info: CanvasInfo,
  xs: number[],
  ys: number[],
  ysStd: number[],
  data: DataPoint[],
  title: string,
  boundary?: number | null,
): void {
  const { canvas, ctx } = info;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width;
  const H = canvas.height;

  const marginTop = 40 * dpr;
  const marginBottom = 36 * dpr;
  const marginLeft = 10 * dpr;
  const marginRight = 10 * dpr;

  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;

  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, W, H);

  const xMin = xs[0]!;
  const xMax = xs[xs.length - 1]!;

  const toPixelX = (x: number): number =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotW;
  const toPixelY = (y: number): number =>
    marginTop + plotH - y * plotH;

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop + plotH);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH);
  ctx.stroke();

  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop);
  ctx.lineTo(marginLeft + plotW, marginTop);
  ctx.stroke();

  // Uncertainty band (mean ± sigma)
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const upper = Math.min(1, Math.max(0, ys[i]! + PREDICTIVE_BAND_SIGMA * ysStd[i]!));
    const px = toPixelX(xs[i]!);
    const py = toPixelY(upper);
    if (i === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  for (let i = xs.length - 1; i >= 0; i--) {
    const lower = Math.min(1, Math.max(0, ys[i]! - PREDICTIVE_BAND_SIGMA * ysStd[i]!));
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(lower));
  }
  ctx.closePath();
  ctx.fillStyle = COLORS.predictiveBand;
  ctx.fill();

  // Mean fill
  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(0));
  for (let i = 0; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.lineTo(toPixelX(xs[xs.length - 1]!), toPixelY(0));
  ctx.closePath();
  ctx.fillStyle = COLORS.predictiveFill;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(ys[0]!));
  for (let i = 1; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.strokeStyle = COLORS.predictive;
  ctx.lineWidth = 2 * dpr;
  ctx.stroke();

  if (boundary !== undefined && boundary !== null) {
    const bx = toPixelX(boundary);
    ctx.save();
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.strokeStyle = COLORS.predictive;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(bx, marginTop);
    ctx.lineTo(bx, marginTop + plotH);
    ctx.stroke();
    ctx.restore();
  }

  for (let i = 0; i < data.length; i++) {
    const point = data[i]!;
    const px = toPixelX(point.x);
    const baseY = point.y === 1 ? 1 : 0;
    const py = toPixelY(baseY) + jitterOffset(i, 6 * dpr);
    ctx.beginPath();
    ctx.arc(px, py, 4 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = point.y === 1 ? COLORS.dataYes : COLORS.dataNo;
    ctx.fill();
  }

  ctx.fillStyle = COLORS.label;
  ctx.font = `bold ${14 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(title, W / 2, 10 * dpr);
}

function dataToPixel(
  x: number,
  y: 0 | 1,
  rect: DOMRect,
): { px: number; py: number } {
  const marginLeft = DATA_STRIP_LAYOUT.marginLeft;
  const marginRight = DATA_STRIP_LAYOUT.marginRight;
  const marginTop = DATA_STRIP_LAYOUT.marginTop;
  const marginBottom = DATA_STRIP_LAYOUT.marginBottom;

  const plotW = rect.width - marginLeft - marginRight;
  const plotH = rect.height - marginTop - marginBottom;

  const px = marginLeft + ((x - X_MIN) / (X_MAX - X_MIN)) * plotW;
  const yYes = marginTop + plotH * DATA_STRIP_LAYOUT.topRow;
  const yNo = marginTop + plotH * DATA_STRIP_LAYOUT.bottomRow;
  const py = y === 1 ? yYes : yNo;

  return { px, py };
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentStep: StepMode = 'likelihood';
let showPredictive = false;
let allowEditInPosterior = false;
let dataPoints: DataPoint[] = DEFAULT_DATA.map((point) => ({ ...point }));
let currentGrid: GridResult = computeGrid(dataPoints);

const dataInfo = getCanvasInfo('data-canvas');
const priorInfo = getCanvasInfo('prior-canvas');
const likelihoodInfo = getCanvasInfo('likelihood-canvas');
const posteriorInfo = getCanvasInfo('posterior-canvas');
const predictiveInfo = getCanvasInfo('predictive-canvas');

// UI elements
const subtitle = document.getElementById('subtitle') as HTMLParagraphElement;
const formulaLabel = document.getElementById('formula-label') as HTMLDivElement;
const dataHint = document.getElementById('data-hint') as HTMLDivElement;
const stepLikelihoodBtn = document.getElementById('step-likelihood') as HTMLButtonElement;
const stepPosteriorBtn = document.getElementById('step-posterior') as HTMLButtonElement;
const predictiveToggle = document.getElementById('toggle-predictive') as HTMLInputElement;
const editToggle = document.getElementById('toggle-edit') as HTMLInputElement;
const resetDataBtn = document.getElementById('reset-data') as HTMLButtonElement;
const clearDataBtn = document.getElementById('clear-data') as HTMLButtonElement;
const presetSeparableBtn = document.getElementById('preset-separable') as HTMLButtonElement;
const presetOverlapBtn = document.getElementById('preset-overlap') as HTMLButtonElement;
const presetContradictBtn = document.getElementById('preset-contradict') as HTMLButtonElement;
const statMode = document.getElementById('stat-mode') as HTMLSpanElement;
const statMean = document.getElementById('stat-mean') as HTMLSpanElement;
const statStd = document.getElementById('stat-std') as HTMLSpanElement;
const statGridPoints = document.getElementById('stat-grid-points') as HTMLSpanElement;
const statDataCount = document.getElementById('stat-data-count') as HTMLSpanElement;
const statBalance = document.getElementById('stat-balance') as HTMLSpanElement;
const statEntropy = document.getElementById('stat-entropy') as HTMLSpanElement;
const statBoundary = document.getElementById('stat-boundary') as HTMLSpanElement;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function computeDataStats(points: DataPoint[]): { balance: number | null; entropy: number | null } {
  if (points.length === 0) {
    return { balance: null, entropy: null };
  }
  const positives = points.reduce((sum, point) => sum + (point.y === 1 ? 1 : 0), 0);
  const balance = positives / points.length;
  if (balance === 0 || balance === 1) {
    return { balance, entropy: 0 };
  }
  const entropy = -balance * Math.log2(balance) - (1 - balance) * Math.log2(1 - balance);
  return { balance, entropy };
}

function updateStats(): void {
  statMode.textContent = currentGrid.posteriorMode.toFixed(3);
  statMean.textContent = currentGrid.posteriorMean.toFixed(3);
  statStd.textContent = currentGrid.posteriorStd.toFixed(3);
  statGridPoints.textContent = String(NUM_GRID_POINTS);
  statDataCount.textContent = String(dataPoints.length);
  const dataStats = computeDataStats(dataPoints);
  statBalance.textContent = dataStats.balance === null ? '-' : `${(dataStats.balance * 100).toFixed(0)}%`;
  statEntropy.textContent = dataStats.entropy === null ? '-' : dataStats.entropy.toFixed(3);
  statBoundary.textContent = currentGrid.decisionBoundary === null
    ? '-'
    : currentGrid.decisionBoundary.toFixed(2);
}

function updateStepCopy(): void {
  if (currentStep === 'likelihood') {
    subtitle.textContent = 'Click to add data points and watch the likelihood emerge.';
    formulaLabel.innerHTML =
      '<span class="formula-highlight">Likelihood</span> from <span class="formula-highlight">Data</span>';
    dataHint.textContent =
      'Click above the midline for y=1, below for y=0. Shift-click a point to remove.';
  } else {
    subtitle.textContent = 'Multiply prior and likelihood to form the posterior, then check predictions.';
    formulaLabel.innerHTML =
      '<span class="formula-highlight">Posterior</span> &prop; <span class="formula-highlight">Prior</span> &times; <span class="formula-highlight">Likelihood</span>';
    dataHint.textContent = allowEditInPosterior
      ? 'Editing enabled in Step 2. Shift-click to remove the nearest point.'
      : 'Data locked in Step 2. Toggle editing if you want to adjust points.';
  }
}

function renderAll(): void {
  resizeCanvas(dataInfo.canvas);
  resizeCanvas(priorInfo.canvas);
  resizeCanvas(likelihoodInfo.canvas);
  resizeCanvas(posteriorInfo.canvas);
  resizeCanvas(predictiveInfo.canvas);

  drawDataStrip(dataInfo, dataPoints);

  drawCurve(
    priorInfo,
    currentGrid.xs,
    currentGrid.prior,
    COLORS.prior,
    COLORS.priorFill,
    'Prior',
  );

  drawCurve(
    likelihoodInfo,
    currentGrid.xs,
    currentGrid.likelihood,
    COLORS.likelihood,
    COLORS.likelihoodFill,
    'Likelihood',
  );

  drawCurve(
    posteriorInfo,
    currentGrid.xs,
    currentGrid.posterior,
    COLORS.posterior,
    COLORS.posteriorFill,
    'Posterior',
    currentGrid.posteriorMode,
  );

  drawPredictive(
    predictiveInfo,
    currentGrid.xs,
    currentGrid.posteriorPredictiveMean,
    currentGrid.posteriorPredictiveStd,
    dataPoints,
    'Posterior Predictive',
    currentGrid.decisionBoundary,
  );

  updateStats();
}

// ---------------------------------------------------------------------------
// Recompute + render
// ---------------------------------------------------------------------------

function recompute(): void {
  currentGrid = computeGrid(dataPoints);
  renderAll();
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

function setStep(step: StepMode): void {
  currentStep = step;
  document.body.dataset.step = step;
  stepLikelihoodBtn.classList.toggle('active', step === 'likelihood');
  stepPosteriorBtn.classList.toggle('active', step === 'posterior');

  if (step !== 'posterior') {
    showPredictive = false;
    predictiveToggle.checked = false;
  }
  predictiveToggle.disabled = step !== 'posterior';
  editToggle.disabled = step !== 'posterior';
  editToggle.checked = allowEditInPosterior;
  if (step !== 'posterior') {
    document.body.dataset.edit = 'on';
  } else {
    document.body.dataset.edit = allowEditInPosterior ? 'on' : 'off';
  }
  document.body.dataset.predictive = showPredictive ? 'on' : 'off';
  updateStepCopy();
  renderAll();
}

function setPredictive(show: boolean): void {
  if (currentStep !== 'posterior') {
    predictiveToggle.checked = false;
    showPredictive = false;
  } else {
    showPredictive = show;
  }
  document.body.dataset.predictive = showPredictive ? 'on' : 'off';
  renderAll();
}

function setAllowEditInPosterior(allow: boolean): void {
  allowEditInPosterior = allow;
  if (currentStep === 'posterior') {
    document.body.dataset.edit = allow ? 'on' : 'off';
  }
  updateStepCopy();
}

stepLikelihoodBtn.addEventListener('click', () => setStep('likelihood'));
stepPosteriorBtn.addEventListener('click', () => setStep('posterior'));

predictiveToggle.addEventListener('change', () => {
  setPredictive(predictiveToggle.checked);
});

editToggle.addEventListener('change', () => {
  setAllowEditInPosterior(editToggle.checked);
});

resetDataBtn.addEventListener('click', () => {
  dataPoints = DEFAULT_DATA.map((point) => ({ ...point }));
  recompute();
});

clearDataBtn.addEventListener('click', () => {
  dataPoints = [];
  recompute();
});

function applyPreset(name: keyof typeof PRESETS): void {
  dataPoints = PRESETS[name].map((point) => ({ ...point }));
  recompute();
}

presetSeparableBtn.addEventListener('click', () => applyPreset('separable'));
presetOverlapBtn.addEventListener('click', () => applyPreset('overlap'));
presetContradictBtn.addEventListener('click', () => applyPreset('contradict'));

function addDataPoint(point: DataPoint): void {
  if (dataPoints.length >= MAX_DATA_POINTS) return;
  dataPoints = [...dataPoints, point];
}

function removeNearestDataPoint(clickX: number, clickY: number, rect: DOMRect): boolean {
  if (dataPoints.length === 0) return false;

  let bestIdx = -1;
  let bestDist = Infinity;
  for (let i = 0; i < dataPoints.length; i++) {
    const point = dataPoints[i]!;
    const { px, py } = dataToPixel(point.x, point.y, rect);
    const dx = px - clickX;
    const dy = py - clickY;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }

  const threshold = 14;
  if (bestIdx >= 0 && bestDist <= threshold * threshold) {
    dataPoints = dataPoints.filter((_, idx) => idx !== bestIdx);
    return true;
  }
  return false;
}

function canEditData(): boolean {
  return currentStep === 'likelihood' || (currentStep === 'posterior' && allowEditInPosterior);
}

dataInfo.canvas.addEventListener('click', (event: MouseEvent) => {
  if (!canEditData()) return;

  const rect = dataInfo.canvas.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;

  if (event.shiftKey) {
    if (removeNearestDataPoint(clickX, clickY, rect)) {
      recompute();
    }
    return;
  }

  const marginLeft = DATA_STRIP_LAYOUT.marginLeft;
  const marginRight = DATA_STRIP_LAYOUT.marginRight;
  const marginTop = DATA_STRIP_LAYOUT.marginTop;
  const marginBottom = DATA_STRIP_LAYOUT.marginBottom;

  const plotW = rect.width - marginLeft - marginRight;
  const plotH = rect.height - marginTop - marginBottom;
  const xNorm = (clickX - marginLeft) / plotW;
  const xValue = X_MIN + Math.min(Math.max(xNorm, 0), 1) * (X_MAX - X_MIN);

  const midY = marginTop + plotH / 2;
  const yValue: 0 | 1 = clickY < midY ? 1 : 0;

  addDataPoint({ x: xValue, y: yValue });
  recompute();
});

window.addEventListener('resize', () => {
  renderAll();
});

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

interface VizDebugState {
  step: StepMode;
  showPredictive: boolean;
  allowEditInPosterior: boolean;
  dataPoints: DataPoint[];
  dataBalance: number | null;
  dataEntropy: number | null;
  decisionBoundary: number | null;
  posteriorMode: number;
  posteriorMean: number;
  posteriorStd: number;
  numGridPoints: number;
}

interface VizDebugAPI {
  getState: () => VizDebugState;
  step: () => VizDebugState;
  reset: () => VizDebugState;
  setConfig: (config: {
    step?: StepMode;
    showPredictive?: boolean;
    allowEditInPosterior?: boolean;
    preset?: keyof typeof PRESETS;
    dataPoints?: DataPoint[];
  }) => VizDebugState;
}

function getDebugState(): VizDebugState {
  const dataStats = computeDataStats(dataPoints);
  return {
    step: currentStep,
    showPredictive,
    allowEditInPosterior,
    dataPoints: dataPoints.map((point) => ({ ...point })),
    dataBalance: dataStats.balance,
    dataEntropy: dataStats.entropy,
    decisionBoundary: currentGrid.decisionBoundary,
    posteriorMode: currentGrid.posteriorMode,
    posteriorMean: currentGrid.posteriorMean,
    posteriorStd: currentGrid.posteriorStd,
    numGridPoints: NUM_GRID_POINTS,
  };
}

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = {
  getState: (): VizDebugState => {
    return getDebugState();
  },

  step: (): VizDebugState => {
    setStep(currentStep === 'likelihood' ? 'posterior' : 'likelihood');
    return getDebugState();
  },

  reset: (): VizDebugState => {
    dataPoints = DEFAULT_DATA.map((point) => ({ ...point }));
    setStep('likelihood');
    showPredictive = false;
    allowEditInPosterior = false;
    predictiveToggle.checked = false;
    editToggle.checked = false;
    document.body.dataset.predictive = 'off';
    recompute();
    return getDebugState();
  },

  setConfig: (config: {
    step?: StepMode;
    showPredictive?: boolean;
    allowEditInPosterior?: boolean;
    preset?: keyof typeof PRESETS;
    dataPoints?: DataPoint[];
  }): VizDebugState => {
    if (config.step) {
      setStep(config.step);
    }
    if (config.showPredictive !== undefined) {
      predictiveToggle.checked = config.showPredictive;
      setPredictive(config.showPredictive);
    }
    if (config.allowEditInPosterior !== undefined) {
      editToggle.checked = config.allowEditInPosterior;
      setAllowEditInPosterior(config.allowEditInPosterior);
    }
    if (config.preset) {
      applyPreset(config.preset);
    }
    if (config.dataPoints) {
      dataPoints = config.dataPoints.map((point) => ({
        x: point.x,
        y: point.y === 1 ? 1 : 0,
      }));
      recompute();
    }
    return getDebugState();
  },
};

console.log('[SHAPES-VIZ] Debug API exposed to window.__vizDebug');

// ---------------------------------------------------------------------------
// Debug command polling loop (only in dev mode)
// ---------------------------------------------------------------------------

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
            result = api.setConfig(cmd.payload as {
              step?: StepMode;
              showPredictive?: boolean;
              dataPoints?: DataPoint[];
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

console.log('[SHAPES-VIZ] Initializing posterior shapes visualization');
setStep('likelihood');
recompute();
console.log('[SHAPES-VIZ] Ready');
