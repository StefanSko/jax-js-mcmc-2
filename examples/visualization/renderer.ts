/**
 * Canvas rendering utilities for HMC visualization.
 */

import type { ContourLine, Bounds } from './contour';

export interface Sample {
  x: number;
  y: number;
  accepted: boolean;
  divergent: boolean;
}

export interface RenderColors {
  accepted: string;
  rejected: string;
  divergent: string;
  current: string;
  contour: string;
  background: string;
  trajectory: string;
}

const DEFAULT_COLORS: RenderColors = {
  accepted: '#4ade80',   // Green
  rejected: '#f87171',   // Red
  divergent: '#facc15',  // Yellow
  current: '#60a5fa',    // Blue
  contour: '#334155',    // Slate
  background: '#16213e', // Dark blue
  trajectory: '#94a3b8', // Gray
};

export class CanvasRenderer {
  private ctx: CanvasRenderingContext2D;
  private width: number = 0;
  private height: number = 0;
  private bounds: Bounds;
  private colors: RenderColors;
  private padding = 40;

  constructor(
    canvas: HTMLCanvasElement,
    bounds: Bounds,
    colors: Partial<RenderColors> = {}
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;
    this.bounds = bounds;
    this.colors = { ...DEFAULT_COLORS, ...colors };
    this.resize();
  }

  /**
   * Update canvas size to match container.
   */
  resize(): void {
    const canvas = this.ctx.canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    this.width = rect.width;
    this.height = rect.height;

    canvas.width = this.width * dpr;
    canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  /**
   * Update visualization bounds.
   */
  setBounds(bounds: Bounds): void {
    this.bounds = bounds;
  }

  /**
   * Transform world coordinates to canvas coordinates.
   */
  private worldToCanvas(x: number, y: number): [number, number] {
    const { xMin, xMax, yMin, yMax } = this.bounds;
    const plotWidth = this.width - 2 * this.padding;
    const plotHeight = this.height - 2 * this.padding;

    const canvasX = this.padding + ((x - xMin) / (xMax - xMin)) * plotWidth;
    // Flip Y axis (canvas Y increases downward)
    const canvasY = this.padding + ((yMax - y) / (yMax - yMin)) * plotHeight;

    return [canvasX, canvasY];
  }

  /**
   * Clear the canvas.
   */
  clear(): void {
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }

  /**
   * Draw axis labels and grid.
   */
  drawAxes(): void {
    const { xMin, xMax, yMin, yMax } = this.bounds;
    const ctx = this.ctx;

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#94a3b8';

    // X axis labels
    const xSteps = 5;
    for (let i = 0; i <= xSteps; i++) {
      const x = xMin + (i / xSteps) * (xMax - xMin);
      const [cx, cy] = this.worldToCanvas(x, yMin);
      ctx.fillText(x.toFixed(1), cx - 12, cy + 16);
    }

    // Y axis labels
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const y = yMin + (i / ySteps) * (yMax - yMin);
      const [cx, cy] = this.worldToCanvas(xMin, y);
      ctx.fillText(y.toFixed(1), cx - 35, cy + 4);
    }
  }

  /**
   * Draw density contour lines.
   */
  drawContours(contours: ContourLine[]): void {
    const ctx = this.ctx;
    ctx.strokeStyle = this.colors.contour;
    ctx.lineWidth = 1;

    for (const contour of contours) {
      const { points } = contour;

      // Draw line segments (pairs of points)
      for (let i = 0; i < points.length; i += 2) {
        const [x1, y1] = points[i];
        const [x2, y2] = points[i + 1];

        const [cx1, cy1] = this.worldToCanvas(x1, y1);
        const [cx2, cy2] = this.worldToCanvas(x2, y2);

        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();
      }
    }
  }

  /**
   * Draw accumulated samples.
   */
  drawSamples(samples: Sample[]): void {
    const ctx = this.ctx;

    for (const sample of samples) {
      const [cx, cy] = this.worldToCanvas(sample.x, sample.y);

      // Choose color based on status
      if (sample.divergent) {
        ctx.fillStyle = this.colors.divergent;
      } else if (sample.accepted) {
        ctx.fillStyle = this.colors.accepted;
      } else {
        ctx.fillStyle = this.colors.rejected;
      }

      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * Draw current position with highlight.
   */
  drawCurrentPosition(x: number, y: number): void {
    const ctx = this.ctx;
    const [cx, cy] = this.worldToCanvas(x, y);

    // Outer glow
    ctx.fillStyle = this.colors.current + '40';  // 25% opacity
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fill();

    // Inner solid
    ctx.fillStyle = this.colors.current;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Draw trajectory jump arrow from previous to current position.
   */
  drawTrajectoryJump(
    fromX: number,
    fromY: number,
    toX: number,
    toY: number
  ): void {
    const ctx = this.ctx;
    const [x1, y1] = this.worldToCanvas(fromX, fromY);
    const [x2, y2] = this.worldToCanvas(toX, toY);

    // Don't draw if same position
    const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    if (dist < 5) return;

    ctx.strokeStyle = this.colors.trajectory;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const headLen = 8;

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - headLen * Math.cos(angle - Math.PI / 6),
      y2 - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      x2 - headLen * Math.cos(angle + Math.PI / 6),
      y2 - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = this.colors.trajectory;
    ctx.fill();
  }

  /**
   * Full render cycle.
   */
  render(
    contours: ContourLine[],
    samples: Sample[],
    currentX: number,
    currentY: number,
    prevX?: number,
    prevY?: number
  ): void {
    this.clear();
    this.drawContours(contours);
    this.drawAxes();
    this.drawSamples(samples);

    if (prevX !== undefined && prevY !== undefined) {
      this.drawTrajectoryJump(prevX, prevY, currentX, currentY);
    }

    this.drawCurrentPosition(currentX, currentY);
  }
}
