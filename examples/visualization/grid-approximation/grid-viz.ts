/**
 * Grid Approximation Visualization
 *
 * Pure JS visualization (no JAX-JS dependency) demonstrating the curse
 * of dimensionality. Shows why grid-based methods fail in high dimensions:
 * the number of evaluations grows as bins^d.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface VizState {
  dimension: 1 | 2 | 3;
  bins: number;
  evaluations: number;
  timeMs: number;
}

interface VizDebugAPI {
  getState: () => VizState;
  step: () => VizState;
  reset: () => VizState;
  setConfig: (config: { bins?: number; dimension?: 1 | 2 | 3 }) => VizState;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DOMAIN_MIN = -4;
const DOMAIN_MAX = 4;
const DOMAIN_RANGE = DOMAIN_MAX - DOMAIN_MIN;

const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
const INV_2PI = 1 / (2 * Math.PI);

const DEFAULT_BINS = 20;

/** Slider range limits per dimension */
const BINS_LIMITS: Record<1 | 2 | 3, { min: number; max: number }> = {
  1: { min: 10, max: 500 },
  2: { min: 10, max: 100 },
  3: { min: 10, max: 1000 },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentDimension: 1 | 2 | 3 = 1;
let currentBins = DEFAULT_BINS;
let lastTimeMs = 0;
let lastEvaluations = 0;
let playInterval: ReturnType<typeof setInterval> | null = null;

// ---------------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------------

const canvas1d = document.getElementById('canvas-1d') as HTMLCanvasElement;
const canvas2d = document.getElementById('canvas-2d') as HTMLCanvasElement;
const ctx1d = canvas1d.getContext('2d')!;
const ctx2d = canvas2d.getContext('2d')!;

const panel1d = document.getElementById('panel-1d')!;
const panel2d = document.getElementById('panel-2d')!;
const panel3d = document.getElementById('panel-3d')!;

const binsSlider = document.getElementById('bins-slider') as HTMLInputElement;
const binsDisplay = document.getElementById('bins-display')!;
const evalCount = document.getElementById('eval-count')!;

const statDim = document.getElementById('stat-dim')!;
const statBins = document.getElementById('stat-bins')!;
const statEvals = document.getElementById('stat-evals')!;
const statTime = document.getElementById('stat-time')!;
const statIntegral = document.getElementById('stat-integral')!;

const scale1d = document.getElementById('scale-1d')!;
const scale2d = document.getElementById('scale-2d')!;
const scale3d = document.getElementById('scale-3d')!;
const scale10d = document.getElementById('scale-10d')!;
const scale100d = document.getElementById('scale-100d')!;

const counterValue = document.getElementById('counter-value')!;
const counterFormula = document.getElementById('counter-formula')!;
const counterWarning = document.getElementById('counter-warning')!;

const insightBox = document.getElementById('insight')!;
const playBtn = document.getElementById('play-btn')!;
const resetBtn = document.getElementById('reset-btn')!;

const dimTabs = document.querySelectorAll<HTMLButtonElement>('.dim-tab');

// ---------------------------------------------------------------------------
// Density functions (pure math, no JAX-JS)
// ---------------------------------------------------------------------------

/** Standard normal density: (1/sqrt(2pi)) * exp(-0.5 * x^2) */
function normalDensity1d(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

/** Standard bivariate normal density: (1/2pi) * exp(-0.5 * (x^2 + y^2)) */
function normalDensity2d(x: number, y: number): number {
  return INV_2PI * Math.exp(-0.5 * (x * x + y * y));
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/** Format number with commas: 1234567 -> "1,234,567" */
function formatWithCommas(n: number): string {
  if (n < 1000) return String(n);
  return n.toLocaleString('en-US');
}

/** Format as exponential with Unicode superscripts: 1.0 x 10^9 */
function formatExponential(n: number): string {
  if (n < 1e6) return formatWithCommas(n);
  const exp = Math.floor(Math.log10(n));
  const mantissa = n / Math.pow(10, exp);

  const superscriptDigits: Record<string, string> = {
    '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
    '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
    '8': '\u2078', '9': '\u2079',
  };

  const supStr = String(exp).split('').map(c => superscriptDigits[c] ?? c).join('');
  return `${mantissa.toFixed(2)} \u00D7 10${supStr}`;
}

/** Format as HTML with <sup> tag for the scaling table */
function formatExponentialHTML(n: number): string {
  if (n < 1e6) return formatWithCommas(n);
  const exp = Math.floor(Math.log10(n));
  const mantissa = n / Math.pow(10, exp);
  return `${mantissa.toFixed(2)} &times; 10<sup>${exp}</sup>`;
}

/** Compute bins^dim, return as number (may be Infinity for large values) */
function computeEvaluations(bins: number, dim: number): number {
  return Math.pow(bins, dim);
}

// ---------------------------------------------------------------------------
// Drawing: 1D grid approximation
// ---------------------------------------------------------------------------

function draw1d(bins: number): { evaluations: number; integral: number; timeMs: number } {
  const t0 = performance.now();

  const w = canvas1d.width;
  const h = canvas1d.height;
  ctx1d.clearRect(0, 0, w, h);

  // Compute grid values
  const dx = DOMAIN_RANGE / bins;
  const gridY: number[] = [];

  for (let i = 0; i < bins; i++) {
    const x = DOMAIN_MIN + (i + 0.5) * dx;
    gridY.push(normalDensity1d(x));
  }

  const evaluations = bins;
  const integral = gridY.reduce((sum, y) => sum + y * dx, 0);

  // Find max density for scaling
  const maxDensity = Math.max(...gridY, normalDensity1d(0));

  // Mapping helpers
  const pad = { left: 60, right: 20, top: 30, bottom: 40 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  function mapX(x: number): number {
    return pad.left + ((x - DOMAIN_MIN) / DOMAIN_RANGE) * plotW;
  }
  function mapY(density: number): number {
    return pad.top + plotH - (density / (maxDensity * 1.15)) * plotH;
  }

  // Draw axes
  ctx1d.strokeStyle = '#334155';
  ctx1d.lineWidth = 1;
  ctx1d.beginPath();
  ctx1d.moveTo(pad.left, pad.top);
  ctx1d.lineTo(pad.left, pad.top + plotH);
  ctx1d.lineTo(pad.left + plotW, pad.top + plotH);
  ctx1d.stroke();

  // Axis labels
  ctx1d.fillStyle = '#64748b';
  ctx1d.font = '11px system-ui';
  ctx1d.textAlign = 'center';
  for (let x = -4; x <= 4; x += 2) {
    const px = mapX(x);
    ctx1d.fillText(String(x), px, pad.top + plotH + 20);

    // Tick marks
    ctx1d.beginPath();
    ctx1d.moveTo(px, pad.top + plotH);
    ctx1d.lineTo(px, pad.top + plotH + 5);
    ctx1d.strokeStyle = '#475569';
    ctx1d.stroke();
  }

  // Y-axis labels
  ctx1d.textAlign = 'right';
  const ySteps = [0, 0.1, 0.2, 0.3, 0.4];
  for (const yVal of ySteps) {
    if (yVal <= maxDensity * 1.15) {
      const py = mapY(yVal);
      ctx1d.fillText(yVal.toFixed(1), pad.left - 8, py + 4);

      // Grid line
      ctx1d.beginPath();
      ctx1d.moveTo(pad.left, py);
      ctx1d.lineTo(pad.left + plotW, py);
      ctx1d.strokeStyle = '#1e293b';
      ctx1d.lineWidth = 0.5;
      ctx1d.stroke();
      ctx1d.lineWidth = 1;
    }
  }

  // Draw grid bars (semi-transparent blue)
  ctx1d.fillStyle = 'rgba(96, 165, 250, 0.35)';
  ctx1d.strokeStyle = 'rgba(96, 165, 250, 0.6)';
  ctx1d.lineWidth = bins > 100 ? 0.5 : 1;

  for (let i = 0; i < bins; i++) {
    const xLeft = DOMAIN_MIN + i * dx;
    const xRight = xLeft + dx;
    const px1 = mapX(xLeft);
    const px2 = mapX(xRight);
    const py = mapY(gridY[i]);
    const baseline = mapY(0);

    ctx1d.fillRect(px1, py, px2 - px1, baseline - py);
    ctx1d.strokeRect(px1, py, px2 - px1, baseline - py);
  }

  // Draw smooth true density curve (white)
  const smoothN = 500;
  ctx1d.beginPath();
  ctx1d.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx1d.lineWidth = 2;

  for (let i = 0; i <= smoothN; i++) {
    const x = DOMAIN_MIN + (i / smoothN) * DOMAIN_RANGE;
    const y = normalDensity1d(x);
    const px = mapX(x);
    const py = mapY(y);
    if (i === 0) ctx1d.moveTo(px, py);
    else ctx1d.lineTo(px, py);
  }
  ctx1d.stroke();

  // Legend
  ctx1d.font = '12px system-ui';
  const legendX = pad.left + 10;
  const legendY = pad.top + 16;

  // Blue square for grid
  ctx1d.fillStyle = 'rgba(96, 165, 250, 0.35)';
  ctx1d.strokeStyle = 'rgba(96, 165, 250, 0.6)';
  ctx1d.fillRect(legendX, legendY - 9, 12, 12);
  ctx1d.strokeRect(legendX, legendY - 9, 12, 12);
  ctx1d.fillStyle = '#94a3b8';
  ctx1d.textAlign = 'left';
  ctx1d.fillText('Grid approximation', legendX + 18, legendY + 1);

  // White line for true density
  ctx1d.beginPath();
  ctx1d.moveTo(legendX, legendY + 15);
  ctx1d.lineTo(legendX + 12, legendY + 15);
  ctx1d.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx1d.lineWidth = 2;
  ctx1d.stroke();
  ctx1d.fillStyle = '#94a3b8';
  ctx1d.fillText('True density', legendX + 18, legendY + 19);

  // Integral annotation
  ctx1d.fillStyle = '#4ade80';
  ctx1d.font = '13px "SF Mono", Monaco, monospace';
  ctx1d.textAlign = 'right';
  ctx1d.fillText(
    `\u222B \u2248 ${integral.toFixed(4)}  (true = 1.0)`,
    pad.left + plotW - 4,
    pad.top + 18,
  );

  const timeMs = performance.now() - t0;
  return { evaluations, integral, timeMs };
}

// ---------------------------------------------------------------------------
// Drawing: 2D heatmap
// ---------------------------------------------------------------------------

/** Purple-blue-white colormap: dark=low, bright=high */
function colormapPBW(t: number): [number, number, number] {
  // t in [0,1]
  // 0.0 -> very dark purple (#0d0221)
  // 0.3 -> deep blue (#1a1a6e)
  // 0.6 -> bright blue (#4488dd)
  // 0.8 -> light cyan (#88ccee)
  // 1.0 -> white (#ffffff)

  if (t <= 0) return [13, 2, 33];
  if (t >= 1) return [255, 255, 255];

  const stops: Array<{ pos: number; r: number; g: number; b: number }> = [
    { pos: 0.0, r: 13, g: 2, b: 33 },
    { pos: 0.15, r: 35, g: 10, b: 80 },
    { pos: 0.3, r: 50, g: 30, b: 120 },
    { pos: 0.5, r: 60, g: 100, b: 180 },
    { pos: 0.7, r: 100, g: 170, b: 220 },
    { pos: 0.85, r: 170, g: 220, b: 240 },
    { pos: 1.0, r: 255, g: 255, b: 255 },
  ];

  // Find the two stops we're between
  let lo = stops[0];
  let hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].pos && t <= stops[i + 1].pos) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }

  const frac = (t - lo.pos) / (hi.pos - lo.pos);
  const r = Math.round(lo.r + frac * (hi.r - lo.r));
  const g = Math.round(lo.g + frac * (hi.g - lo.g));
  const b = Math.round(lo.b + frac * (hi.b - lo.b));
  return [r, g, b];
}

function draw2d(bins: number): { evaluations: number; integral: number; timeMs: number } {
  const t0 = performance.now();

  const w = canvas2d.width;
  const h = canvas2d.height;
  ctx2d.clearRect(0, 0, w, h);

  const evaluations = bins * bins;
  const dx = DOMAIN_RANGE / bins;

  // Compute grid of densities
  const densities = new Float64Array(bins * bins);
  let maxDensity = 0;
  let integral = 0;

  for (let iy = 0; iy < bins; iy++) {
    const y = DOMAIN_MIN + (iy + 0.5) * dx;
    for (let ix = 0; ix < bins; ix++) {
      const x = DOMAIN_MIN + (ix + 0.5) * dx;
      const d = normalDensity2d(x, y);
      densities[iy * bins + ix] = d;
      if (d > maxDensity) maxDensity = d;
      integral += d * dx * dx;
    }
  }

  // Layout: leave room for axis labels
  const pad = { left: 50, right: 20, top: 30, bottom: 40 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  // Draw heatmap using ImageData for efficiency
  const imageData = ctx2d.createImageData(w, h);
  const data = imageData.data;

  // Fill background
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 22;     // #16213e
    data[i + 1] = 33;
    data[i + 2] = 62;
    data[i + 3] = 255;
  }

  // Map each pixel in the plot area to the closest grid cell
  for (let py = 0; py < plotH; py++) {
    const screenY = pad.top + py;
    // Flip Y so that y increases upward
    const normY = 1 - py / plotH;
    const gridIy = Math.floor(normY * bins);
    const clampedIy = Math.min(Math.max(gridIy, 0), bins - 1);

    for (let px = 0; px < plotW; px++) {
      const screenX = pad.left + px;
      const normX = px / plotW;
      const gridIx = Math.floor(normX * bins);
      const clampedIx = Math.min(Math.max(gridIx, 0), bins - 1);

      const d = densities[clampedIy * bins + clampedIx];
      const t = maxDensity > 0 ? d / maxDensity : 0;
      const [r, g, b] = colormapPBW(t);

      const idx = (screenY * w + screenX) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }

  ctx2d.putImageData(imageData, 0, 0);

  // Draw grid lines if bins are few enough to show them
  if (bins <= 50) {
    ctx2d.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx2d.lineWidth = 0.5;

    for (let i = 0; i <= bins; i++) {
      const frac = i / bins;

      // Vertical line
      const px = pad.left + frac * plotW;
      ctx2d.beginPath();
      ctx2d.moveTo(px, pad.top);
      ctx2d.lineTo(px, pad.top + plotH);
      ctx2d.stroke();

      // Horizontal line
      const py = pad.top + frac * plotH;
      ctx2d.beginPath();
      ctx2d.moveTo(pad.left, py);
      ctx2d.lineTo(pad.left + plotW, py);
      ctx2d.stroke();
    }
  }

  // Axes
  ctx2d.strokeStyle = '#334155';
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(pad.left, pad.top);
  ctx2d.lineTo(pad.left, pad.top + plotH);
  ctx2d.lineTo(pad.left + plotW, pad.top + plotH);
  ctx2d.stroke();

  // Axis labels
  ctx2d.fillStyle = '#64748b';
  ctx2d.font = '11px system-ui';
  ctx2d.textAlign = 'center';
  for (let x = -4; x <= 4; x += 2) {
    const px = pad.left + ((x - DOMAIN_MIN) / DOMAIN_RANGE) * plotW;
    ctx2d.fillText(String(x), px, pad.top + plotH + 20);
  }
  ctx2d.textAlign = 'right';
  for (let y = -4; y <= 4; y += 2) {
    const py = pad.top + plotH - ((y - DOMAIN_MIN) / DOMAIN_RANGE) * plotH;
    ctx2d.fillText(String(y), pad.left - 8, py + 4);
  }

  // Axis titles
  ctx2d.fillStyle = '#94a3b8';
  ctx2d.font = '12px system-ui';
  ctx2d.textAlign = 'center';
  ctx2d.fillText('x', pad.left + plotW / 2, pad.top + plotH + 34);

  ctx2d.save();
  ctx2d.translate(14, pad.top + plotH / 2);
  ctx2d.rotate(-Math.PI / 2);
  ctx2d.fillText('y', 0, 0);
  ctx2d.restore();

  // Integral annotation
  ctx2d.fillStyle = '#4ade80';
  ctx2d.font = '13px "SF Mono", Monaco, monospace';
  ctx2d.textAlign = 'right';
  ctx2d.fillText(
    `\u222B \u2248 ${integral.toFixed(4)}  (true = 1.0)`,
    pad.left + plotW - 4,
    pad.top + 18,
  );

  // Colorbar
  const cbX = pad.left + plotW + 6;
  const cbW = 12;
  const cbH = plotH;
  for (let py = 0; py < cbH; py++) {
    const t = 1 - py / cbH;
    const [r, g, b] = colormapPBW(t);
    ctx2d.fillStyle = `rgb(${r},${g},${b})`;
    ctx2d.fillRect(cbX, pad.top + py, cbW, 1);
  }

  const timeMs = performance.now() - t0;
  return { evaluations, integral, timeMs };
}

// ---------------------------------------------------------------------------
// 3D counter (no computation)
// ---------------------------------------------------------------------------

function draw3d(bins: number): { evaluations: number; integral: number; timeMs: number } {
  const t0 = performance.now();
  const evaluations = computeEvaluations(bins, 3);

  counterFormula.innerHTML = `${bins}<sup>3</sup>`;
  counterValue.textContent = evaluations < 1e15
    ? formatWithCommas(evaluations)
    : formatExponential(evaluations);

  // Warning thresholds
  if (evaluations > 1e9) {
    counterWarning.textContent =
      'At this resolution, a 3D grid requires over a billion evaluations. ' +
      'In 10D with the same bins, it would be ' +
      formatExponential(computeEvaluations(bins, 10)) + ' evaluations.';
    counterWarning.style.color = '#f87171';
  } else if (evaluations > 1e6) {
    counterWarning.textContent =
      'Already millions of evaluations for just 3 dimensions. ' +
      'Each added dimension multiplies the cost by another factor of ' + bins + '.';
    counterWarning.style.color = '#fbbf24';
  } else {
    counterWarning.textContent =
      'Manageable in 3D, but watch what happens as you increase bins. ' +
      'Real posteriors live in 10s to 1000s of dimensions.';
    counterWarning.style.color = '#94a3b8';
  }

  const timeMs = performance.now() - t0;
  // No actual integral computation for 3D
  return { evaluations, integral: NaN, timeMs };
}

// ---------------------------------------------------------------------------
// Resize handling
// ---------------------------------------------------------------------------

function resizeCanvas(canvas: HTMLCanvasElement): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  // Reset the stored dimensions used by drawing routines
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resizeAll(): void {
  resizeCanvas(canvas1d);
  resizeCanvas(canvas2d);
}

// ---------------------------------------------------------------------------
// Render dispatch
// ---------------------------------------------------------------------------

function render(): void {
  let result: { evaluations: number; integral: number; timeMs: number };

  switch (currentDimension) {
    case 1:
      resizeCanvas(canvas1d);
      result = draw1d(currentBins);
      break;
    case 2:
      resizeCanvas(canvas2d);
      result = draw2d(currentBins);
      break;
    case 3:
      result = draw3d(currentBins);
      break;
  }

  lastTimeMs = result!.timeMs;
  lastEvaluations = result!.evaluations;

  updateStats(result!);
}

// ---------------------------------------------------------------------------
// UI update helpers
// ---------------------------------------------------------------------------

function updateStats(result: { evaluations: number; integral: number; timeMs: number }): void {
  statDim.textContent = `${currentDimension}D`;
  statBins.textContent = String(currentBins);
  statEvals.textContent = result.evaluations < 1e12
    ? formatWithCommas(result.evaluations)
    : formatExponential(result.evaluations);
  statTime.textContent = result.timeMs < 1
    ? `${(result.timeMs * 1000).toFixed(0)}\u00B5s`
    : `${result.timeMs.toFixed(1)}ms`;
  statIntegral.textContent = Number.isNaN(result.integral)
    ? 'N/A (3D)'
    : result.integral.toFixed(4);

  // Update evaluation banner
  evalCount.textContent = result.evaluations < 1e12
    ? formatWithCommas(result.evaluations)
    : formatExponential(result.evaluations);

  // Update scaling table
  updateScalingTable();

  // Update insight text
  updateInsight();
}

function updateScalingTable(): void {
  const b = currentBins;
  scale1d.textContent = formatWithCommas(b);
  scale2d.textContent = formatWithCommas(b * b);

  const e3 = computeEvaluations(b, 3);
  scale3d.innerHTML = e3 < 1e6 ? formatWithCommas(e3) : formatExponentialHTML(e3);

  const e10 = computeEvaluations(b, 10);
  scale10d.innerHTML = e10 < 1e6 ? formatWithCommas(e10) : formatExponentialHTML(e10);

  const e100 = computeEvaluations(b, 100);
  scale100d.innerHTML = formatExponentialHTML(e100);
}

function updateInsight(): void {
  const b = currentBins;

  if (currentDimension === 1) {
    if (b <= 30) {
      insightBox.innerHTML =
        '<strong>1D is easy:</strong> ' + b + ' bins gives a reasonable ' +
        'approximation. The integral is close to 1.0.';
    } else if (b <= 100) {
      insightBox.innerHTML =
        '<strong>1D is still cheap:</strong> ' + b + ' evaluations and the ' +
        'approximation is very accurate. But switch to 2D...';
    } else {
      insightBox.innerHTML =
        '<strong>Overkill in 1D:</strong> ' + b + ' bins is more than enough. ' +
        'The real question is what happens when you add dimensions.';
    }
  } else if (currentDimension === 2) {
    const evals = b * b;
    if (evals <= 1000) {
      insightBox.innerHTML =
        '<strong>2D is manageable:</strong> ' + formatWithCommas(evals) +
        ' evaluations. You can see the heatmap cells. But we only have 2 parameters.';
    } else if (evals <= 10000) {
      insightBox.innerHTML =
        '<strong>Getting expensive:</strong> ' + formatWithCommas(evals) +
        ' evaluations just for 2D. Imagine a model with 10 parameters...';
    } else {
      insightBox.innerHTML =
        '<strong>Thousands of evaluations:</strong> ' + formatWithCommas(evals) +
        ' grid cells for just 2 dimensions. In 10D this would be ' +
        formatExponential(computeEvaluations(b, 10)) + ' evaluations.';
    }
  } else {
    const evals = computeEvaluations(b, 3);
    if (evals <= 1e6) {
      insightBox.innerHTML =
        '<strong>3D is a stretch:</strong> ' + formatWithCommas(evals) +
        ' evaluations. Feasible on a computer, but posteriors often have ' +
        '10, 100, or 1000+ dimensions.';
    } else if (evals <= 1e9) {
      insightBox.innerHTML =
        '<strong>Millions of evaluations:</strong> ' + formatExponential(evals) +
        '. Each density evaluation might involve a neural network forward pass or ' +
        'ODE solve. This is already impractical.';
    } else {
      insightBox.innerHTML =
        '<strong>The curse of dimensionality:</strong> ' + formatExponential(evals) +
        ' evaluations. No computer can enumerate this grid. ' +
        '<em>This is why we need MCMC.</em>';
    }
  }
}

function switchDimension(dim: 1 | 2 | 3): void {
  currentDimension = dim;

  // Update tabs
  dimTabs.forEach(tab => {
    tab.classList.toggle('active', Number(tab.dataset.dim) === dim);
  });

  // Update panels
  panel1d.classList.toggle('active', dim === 1);
  panel2d.classList.toggle('active', dim === 2);
  panel3d.classList.toggle('active', dim === 3);

  // Update slider range
  const limits = BINS_LIMITS[dim];
  binsSlider.min = String(limits.min);
  binsSlider.max = String(limits.max);

  // Clamp current bins to new range
  if (currentBins < limits.min) currentBins = limits.min;
  if (currentBins > limits.max) currentBins = limits.max;
  binsSlider.value = String(currentBins);
  binsDisplay.textContent = String(currentBins);

  render();
}

function setBins(bins: number): void {
  const limits = BINS_LIMITS[currentDimension];
  currentBins = Math.max(limits.min, Math.min(limits.max, bins));
  binsSlider.value = String(currentBins);
  binsDisplay.textContent = String(currentBins);
  render();
}

// ---------------------------------------------------------------------------
// Play / Reset
// ---------------------------------------------------------------------------

function startPlay(): void {
  if (playInterval !== null) return;

  playBtn.textContent = 'Pause';
  playInterval = setInterval(() => {
    const limits = BINS_LIMITS[currentDimension];
    const newBins = currentBins + 10;
    if (newBins > limits.max) {
      stopPlay();
      return;
    }
    setBins(newBins);
  }, 200);
}

function stopPlay(): void {
  if (playInterval !== null) {
    clearInterval(playInterval);
    playInterval = null;
  }
  playBtn.textContent = 'Play';
}

function togglePlay(): void {
  if (playInterval !== null) {
    stopPlay();
  } else {
    startPlay();
  }
}

function resetViz(): void {
  stopPlay();
  currentBins = DEFAULT_BINS;
  binsSlider.value = String(DEFAULT_BINS);
  binsDisplay.textContent = String(DEFAULT_BINS);
  render();
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

dimTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const dim = Number(tab.dataset.dim) as 1 | 2 | 3;
    stopPlay();
    switchDimension(dim);
  });
});

binsSlider.addEventListener('input', () => {
  const val = Number(binsSlider.value);
  currentBins = val;
  binsDisplay.textContent = String(val);
  render();
});

playBtn.addEventListener('click', togglePlay);
resetBtn.addEventListener('click', resetViz);

window.addEventListener('resize', () => {
  render();
});

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

function buildState(): VizState {
  return {
    dimension: currentDimension,
    bins: currentBins,
    evaluations: lastEvaluations,
    timeMs: lastTimeMs,
  };
}

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = {
  getState(): VizState {
    return buildState();
  },

  step(): VizState {
    const limits = BINS_LIMITS[currentDimension];
    const newBins = Math.min(currentBins + 10, limits.max);
    setBins(newBins);
    return buildState();
  },

  reset(): VizState {
    resetViz();
    return buildState();
  },

  setConfig(config: { bins?: number; dimension?: 1 | 2 | 3 }): VizState {
    if (config.dimension !== undefined) {
      switchDimension(config.dimension);
    }
    if (config.bins !== undefined) {
      setBins(config.bins);
    }
    return buildState();
  },
};

console.log('[GRID-VIZ] Debug API exposed to window.__vizDebug');

// Debug command polling loop (only in dev mode)
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
            result = api.setConfig(cmd.payload as { bins?: number; dimension?: 1 | 2 | 3 });
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
// Bootstrap
// ---------------------------------------------------------------------------

resizeAll();
render();

console.log('[GRID-VIZ] Initialized: dimension=%d, bins=%d', currentDimension, currentBins);
