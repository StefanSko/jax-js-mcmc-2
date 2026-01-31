/**
 * Posterior Shapes Visualization
 *
 * Pure JS visualization showing how Prior x Likelihood -> complex posteriors.
 * No JAX-JS dependency, no MCMC — just 1D density evaluation on a grid.
 */

// Console bridge must be first - sends browser logs to terminal
import '../console-bridge';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type LikelihoodType = 'gaussian' | 'logistic' | 'mixture';

interface GridResult {
  xs: number[];
  prior: number[];
  likelihood: number[];
  posterior: number[];
  posteriorMode: number;
  posteriorIsMultimodal: boolean;
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
  // Numerically stable sigmoid
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

// ---------------------------------------------------------------------------
// Density functions
// ---------------------------------------------------------------------------

/** Standard normal prior N(0, 1) */
function prior(x: number): number {
  return gaussianPdf(x, 0, 1);
}

/** Likelihood functions indexed by type */
const likelihoods: Record<LikelihoodType, (x: number) => number> = {
  /** Gaussian likelihood N(2, 0.5^2) — conjugate case */
  gaussian(x: number): number {
    return gaussianPdf(x, 2, 0.5);
  },

  /** Logistic likelihood sigmoid(3*x) centered at 1.5 */
  logistic(x: number): number {
    return sigmoid(3 * x);
  },

  /** Mixture of two Gaussians: 0.5 * N(-1.5, 0.4^2) + 0.5 * N(1.5, 0.4^2) */
  mixture(x: number): number {
    return 0.5 * gaussianPdf(x, -1.5, 0.4) + 0.5 * gaussianPdf(x, 1.5, 0.4);
  },
};

// ---------------------------------------------------------------------------
// Grid computation
// ---------------------------------------------------------------------------

const NUM_GRID_POINTS = 1000;
const X_MIN = -5;
const X_MAX = 5;

function computeGrid(likelihoodType: LikelihoodType): GridResult {
  const dx = (X_MAX - X_MIN) / (NUM_GRID_POINTS - 1);
  const xs: number[] = [];
  const priorVals: number[] = [];
  const likelihoodVals: number[] = [];
  const unnormalized: number[] = [];

  const likFn = likelihoods[likelihoodType];

  // Evaluate on grid
  for (let i = 0; i < NUM_GRID_POINTS; i++) {
    const x = X_MIN + i * dx;
    xs.push(x);

    const p = prior(x);
    const l = likFn(x);
    priorVals.push(p);
    likelihoodVals.push(l);
    unnormalized.push(p * l);
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

  // Detect multimodality: count local maxima
  const posteriorIsMultimodal = countLocalMaxima(posteriorVals) > 1;

  return {
    xs,
    prior: priorVals,
    likelihood: likelihoodVals,
    posterior: posteriorVals,
    posteriorMode,
    posteriorIsMultimodal,
  };
}

/**
 * Count local maxima in a 1D array.
 * A local maximum is a point higher than both neighbors, with a minimum
 * prominence threshold to filter noise.
 */
function countLocalMaxima(values: number[]): number {
  const n = values.length;
  if (n < 3) return 0;

  // Find the global max for prominence threshold
  let globalMax = -Infinity;
  for (let i = 0; i < n; i++) {
    if (values[i]! > globalMax) globalMax = values[i]!;
  }

  // A peak must be at least 5% of the global max to count
  const prominenceThreshold = globalMax * 0.05;
  let count = 0;

  for (let i = 1; i < n - 1; i++) {
    if (values[i]! > values[i - 1]! && values[i]! > values[i + 1]!) {
      if (values[i]! > prominenceThreshold) {
        count++;
      }
    }
  }

  return count;
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

/**
 * Draw a filled density curve on a canvas.
 *
 * @param info - Canvas and context
 * @param xs - X grid values
 * @param ys - Y density values
 * @param strokeColor - Curve line color
 * @param fillColor - Fill under curve color (with alpha)
 * @param title - Label at top of canvas
 * @param modeLine - If provided, draw a vertical dashed line at this X value
 */
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

  // Margins (in physical pixels)
  const marginTop = 40 * dpr;
  const marginBottom = 36 * dpr;
  const marginLeft = 10 * dpr;
  const marginRight = 10 * dpr;

  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;

  // Clear
  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, W, H);

  // Find Y range
  let yMax = -Infinity;
  for (let i = 0; i < ys.length; i++) {
    if (ys[i]! > yMax) yMax = ys[i]!;
  }
  if (yMax <= 0) yMax = 1; // fallback
  yMax *= 1.1; // 10% headroom

  const xMin = xs[0]!;
  const xMax = xs[xs.length - 1]!;

  // Coordinate mapping
  const toPixelX = (x: number): number =>
    marginLeft + ((x - xMin) / (xMax - xMin)) * plotW;
  const toPixelY = (y: number): number =>
    marginTop + plotH - (y / yMax) * plotH;

  // Draw X axis line
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop + plotH);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH);
  ctx.stroke();

  // X axis tick marks
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

  // Draw filled area under curve
  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(0));
  for (let i = 0; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.lineTo(toPixelX(xs[xs.length - 1]!), toPixelY(0));
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Draw curve line
  ctx.beginPath();
  ctx.moveTo(toPixelX(xs[0]!), toPixelY(ys[0]!));
  for (let i = 1; i < xs.length; i++) {
    ctx.lineTo(toPixelX(xs[i]!), toPixelY(ys[i]!));
  }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2 * dpr;
  ctx.stroke();

  // Draw mode line (vertical dashed)
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

    // Mode value label
    ctx.fillStyle = COLORS.modeLine;
    ctx.font = `${10 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`mode=${modeLine.toFixed(2)}`, mx, marginTop - 2 * dpr);
  }

  // Title label
  ctx.fillStyle = COLORS.label;
  ctx.font = `bold ${14 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(title, W / 2, 10 * dpr);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentLikelihood: LikelihoodType = 'gaussian';
let currentGrid: GridResult = computeGrid(currentLikelihood);

const priorInfo = getCanvasInfo('prior-canvas');
const likelihoodInfo = getCanvasInfo('likelihood-canvas');
const posteriorInfo = getCanvasInfo('posterior-canvas');

// UI elements
const likelihoodSelect = document.getElementById('likelihood') as HTMLSelectElement;
const resetBtn = document.getElementById('reset') as HTMLButtonElement;
const statMode = document.getElementById('stat-mode') as HTMLSpanElement;
const statMultimodal = document.getElementById('stat-multimodal') as HTMLSpanElement;
const statGridPoints = document.getElementById('stat-grid-points') as HTMLSpanElement;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function updateStats(): void {
  statMode.textContent = currentGrid.posteriorMode.toFixed(3);
  statMultimodal.textContent = currentGrid.posteriorIsMultimodal ? 'Yes' : 'No';
  statGridPoints.textContent = String(NUM_GRID_POINTS);
}

function renderAll(): void {
  resizeCanvas(priorInfo.canvas);
  resizeCanvas(likelihoodInfo.canvas);
  resizeCanvas(posteriorInfo.canvas);

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

  updateStats();
}

// ---------------------------------------------------------------------------
// Recompute + render
// ---------------------------------------------------------------------------

function recompute(): void {
  currentGrid = computeGrid(currentLikelihood);
  renderAll();
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

likelihoodSelect.addEventListener('change', () => {
  currentLikelihood = likelihoodSelect.value as LikelihoodType;
  recompute();
});

resetBtn.addEventListener('click', () => {
  currentLikelihood = 'gaussian';
  likelihoodSelect.value = 'gaussian';
  recompute();
});

window.addEventListener('resize', () => {
  renderAll();
});

// ---------------------------------------------------------------------------
// Debug API
// ---------------------------------------------------------------------------

interface VizDebugState {
  likelihood: string;
  posteriorMode: number;
  posteriorIsMultimodal: boolean;
  numGridPoints: number;
}

interface VizDebugAPI {
  getState: () => VizDebugState;
  step: () => VizDebugState;
  reset: () => VizDebugState;
  setConfig: (config: { likelihood?: LikelihoodType }) => VizDebugState;
}

function getDebugState(): VizDebugState {
  return {
    likelihood: currentLikelihood,
    posteriorMode: currentGrid.posteriorMode,
    posteriorIsMultimodal: currentGrid.posteriorIsMultimodal,
    numGridPoints: NUM_GRID_POINTS,
  };
}

const LIKELIHOOD_ORDER: LikelihoodType[] = ['gaussian', 'logistic', 'mixture'];

(window as unknown as { __vizDebug: VizDebugAPI }).__vizDebug = {
  getState: (): VizDebugState => {
    return getDebugState();
  },

  step: (): VizDebugState => {
    // Cycle to next likelihood
    const currentIdx = LIKELIHOOD_ORDER.indexOf(currentLikelihood);
    const nextIdx = (currentIdx + 1) % LIKELIHOOD_ORDER.length;
    currentLikelihood = LIKELIHOOD_ORDER[nextIdx]!;
    likelihoodSelect.value = currentLikelihood;
    recompute();
    return getDebugState();
  },

  reset: (): VizDebugState => {
    currentLikelihood = 'gaussian';
    likelihoodSelect.value = 'gaussian';
    recompute();
    return getDebugState();
  },

  setConfig: (config: { likelihood?: LikelihoodType }): VizDebugState => {
    if (config.likelihood !== undefined && config.likelihood in likelihoods) {
      currentLikelihood = config.likelihood;
      likelihoodSelect.value = currentLikelihood;
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
            result = api.setConfig(cmd.payload as { likelihood?: LikelihoodType });
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
recompute();
console.log('[SHAPES-VIZ] Ready');
