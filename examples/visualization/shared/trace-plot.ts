/**
 * TracePlot — Canvas-based time-series plot.
 *
 * Renders a scrolling line chart of scalar values over time.
 * Used by: rwm-explorer (x-coordinate trace), energy-conservation (H over steps),
 * hmc-explorer (energy error).
 */

export interface TracePlotConfig {
  yLabel: string;
  color: string;
  /** Maximum number of data points before old ones are dropped. Default: 500 */
  maxPoints?: number;
  /** Whether to draw a horizontal reference line. */
  referenceLine?: { value: number; color: string; label?: string };
  /** Background color. Default: #16213e */
  background?: string;
  /** Axis/text color. Default: #94a3b8 */
  axisColor?: string;
}

export class TracePlot {
  private ctx: CanvasRenderingContext2D;
  private data: number[] = [];
  private maxPoints: number;
  private yLabel: string;
  private color: string;
  private referenceLine?: { value: number; color: string; label?: string };
  private background: string;
  private axisColor: string;
  private padding = { top: 20, right: 15, bottom: 30, left: 55 };

  constructor(canvas: HTMLCanvasElement, config: TracePlotConfig) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;
    this.yLabel = config.yLabel;
    this.color = config.color;
    this.maxPoints = config.maxPoints ?? 500;
    this.referenceLine = config.referenceLine;
    this.background = config.background ?? '#16213e';
    this.axisColor = config.axisColor ?? '#94a3b8';
    this.resize();
  }

  /** Resize canvas to match container. Call on window resize. */
  resize(): void {
    const canvas = this.ctx.canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  /** Add a data point. Drops oldest if over maxPoints. */
  addPoint(value: number): void {
    this.data.push(value);
    if (this.data.length > this.maxPoints) {
      this.data.shift();
    }
  }

  /** Replace all data with a new array. */
  setData(values: number[]): void {
    this.data = values.slice(-this.maxPoints);
  }

  /** Clear all data. */
  clear(): void {
    this.data = [];
  }

  /** Update the reference line value (e.g. initial energy H₀). */
  setReferenceLine(ref: { value: number; color: string; label?: string }): void {
    this.referenceLine = ref;
  }

  /** Get current data length. */
  get length(): number {
    return this.data.length;
  }

  /** Render the plot. */
  render(): void {
    const ctx = this.ctx;
    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    const { top, right, bottom, left } = this.padding;
    const plotW = w - left - right;
    const plotH = h - top - bottom;

    // Clear
    ctx.fillStyle = this.background;
    ctx.fillRect(0, 0, w, h);

    if (this.data.length === 0) {
      ctx.fillStyle = this.axisColor;
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No data', w / 2, h / 2);
      return;
    }

    // Compute Y range
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const v of this.data) {
      if (isFinite(v)) {
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
      }
    }
    if (this.referenceLine) {
      yMin = Math.min(yMin, this.referenceLine.value);
      yMax = Math.max(yMax, this.referenceLine.value);
    }
    // Add 10% padding to y range
    const yPad = Math.max((yMax - yMin) * 0.1, 0.01);
    yMin -= yPad;
    yMax += yPad;

    // Helper: data → canvas coords
    const toX = (i: number) => left + (i / Math.max(this.data.length - 1, 1)) * plotW;
    const toY = (v: number) => top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

    // Draw reference line
    if (this.referenceLine) {
      const ry = toY(this.referenceLine.value);
      ctx.strokeStyle = this.referenceLine.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(left, ry);
      ctx.lineTo(left + plotW, ry);
      ctx.stroke();
      ctx.setLineDash([]);

      if (this.referenceLine.label) {
        ctx.fillStyle = this.referenceLine.color;
        ctx.font = '10px system-ui';
        ctx.textAlign = 'right';
        ctx.fillText(this.referenceLine.label, left + plotW, ry - 4);
      }
    }

    // Draw data line
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < this.data.length; i++) {
      const v = this.data[i];
      if (!isFinite(v)) continue;
      const x = toX(i);
      const y = toY(v);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw axes
    ctx.strokeStyle = this.axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, top + plotH);
    ctx.lineTo(left + plotW, top + plotH);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = this.axisColor;
    ctx.font = '10px system-ui';
    ctx.textAlign = 'right';
    const nYTicks = 5;
    for (let i = 0; i <= nYTicks; i++) {
      const v = yMin + (i / nYTicks) * (yMax - yMin);
      const y = toY(v);
      ctx.fillText(v.toFixed(2), left - 4, y + 3);
      // Tick mark
      ctx.beginPath();
      ctx.moveTo(left - 2, y);
      ctx.lineTo(left, y);
      ctx.stroke();
    }

    // X-axis label (step count)
    ctx.fillStyle = this.axisColor;
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Step', left + plotW / 2, h - 4);

    // Y-axis label (rotated)
    ctx.save();
    ctx.translate(12, top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(this.yLabel, 0, 0);
    ctx.restore();
  }
}
