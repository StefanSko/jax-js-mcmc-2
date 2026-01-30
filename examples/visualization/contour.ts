/**
 * Density contour computation using marching squares algorithm.
 * Pure JavaScript implementation for visualization.
 */

export interface Bounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface ContourLine {
  level: number;
  points: Array<[number, number]>;
}

/**
 * Compute density values on a grid.
 * Uses regular JavaScript evaluation (not JAX-JS) for speed.
 */
export function computeDensityGrid(
  logdensityFn: (x: number, y: number) => number,
  bounds: Bounds,
  resolution: number
): { grid: number[][]; xs: number[]; ys: number[] } {
  const { xMin, xMax, yMin, yMax } = bounds;
  const xs: number[] = [];
  const ys: number[] = [];
  const grid: number[][] = [];

  const stepX = (xMax - xMin) / (resolution - 1);
  const stepY = (yMax - yMin) / (resolution - 1);

  for (let i = 0; i < resolution; i++) {
    xs.push(xMin + i * stepX);
    ys.push(yMin + i * stepY);
  }

  for (let j = 0; j < resolution; j++) {
    const row: number[] = [];
    for (let i = 0; i < resolution; i++) {
      // Use exp(logdensity) for actual density values
      const logDensity = logdensityFn(xs[i], ys[j]);
      row.push(logDensity);
    }
    grid.push(row);
  }

  return { grid, xs, ys };
}

/**
 * Compute contour levels automatically based on density range.
 */
export function computeContourLevels(
  grid: number[][],
  numLevels: number = 8
): number[] {
  // Find min/max log-density
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

  if (!isFinite(minVal) || !isFinite(maxVal)) {
    return [];
  }

  // Clamp extreme ranges so heavy-tailed distributions render useful contours.
  // Neal's funnel can span tens of thousands of log units across the grid.
  const maxLogDrop = 30;
  const effectiveMin = Math.max(minVal, maxVal - maxLogDrop);

  // Create evenly spaced levels (in log space)
  const levels: number[] = [];
  const range = maxVal - effectiveMin;

  // Start from near the max (high density) going down
  for (let i = 1; i <= numLevels; i++) {
    const level = maxVal - (i / (numLevels + 1)) * range;
    levels.push(level);
  }

  return levels;
}

/**
 * Marching squares lookup table.
 * Each cell has 4 corners (binary: TL TR BR BL) -> 16 cases.
 * Returns edge pairs to draw: edges 0=top, 1=right, 2=bottom, 3=left.
 */
const MARCHING_SQUARES_EDGES: Record<number, number[][]> = {
  0: [],
  1: [[3, 2]],
  2: [[2, 1]],
  3: [[3, 1]],
  4: [[1, 0]],
  5: [[3, 0], [1, 2]],
  6: [[2, 0]],
  7: [[3, 0]],
  8: [[0, 3]],
  9: [[0, 2]],
  10: [[0, 1], [2, 3]],
  11: [[0, 1]],
  12: [[1, 3]],
  13: [[1, 2]],
  14: [[2, 3]],
  15: [],
};

/**
 * Interpolate position along an edge.
 */
function interpolate(
  v1: number,
  v2: number,
  p1: number,
  p2: number,
  level: number
): number {
  if (Math.abs(v2 - v1) < 1e-10) return (p1 + p2) / 2;
  const t = (level - v1) / (v2 - v1);
  return p1 + t * (p2 - p1);
}

/**
 * Extract contour lines at a given level using marching squares.
 */
export function extractContourAtLevel(
  grid: number[][],
  xs: number[],
  ys: number[],
  level: number
): ContourLine {
  const points: Array<[number, number]> = [];
  const ny = grid.length;
  const nx = grid[0].length;

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      // Cell corners: TL, TR, BR, BL
      const tl = grid[j + 1][i];
      const tr = grid[j + 1][i + 1];
      const br = grid[j][i + 1];
      const bl = grid[j][i];

      // Binary case index
      const caseIndex =
        (tl >= level ? 8 : 0) |
        (tr >= level ? 4 : 0) |
        (br >= level ? 2 : 0) |
        (bl >= level ? 1 : 0);

      const edges = MARCHING_SQUARES_EDGES[caseIndex];
      if (!edges || edges.length === 0) continue;

      // Get cell coordinates
      const x0 = xs[i];
      const x1 = xs[i + 1];
      const y0 = ys[j];
      const y1 = ys[j + 1];

      // Edge midpoint coordinates
      const edgeCoords = [
        [interpolate(tl, tr, x0, x1, level), y1], // top
        [x1, interpolate(tr, br, y1, y0, level)], // right
        [interpolate(bl, br, x0, x1, level), y0], // bottom
        [x0, interpolate(tl, bl, y1, y0, level)], // left
      ];

      for (const [e1, e2] of edges) {
        points.push(edgeCoords[e1] as [number, number]);
        points.push(edgeCoords[e2] as [number, number]);
      }
    }
  }

  return { level, points };
}

/**
 * Extract all contour lines for multiple levels.
 */
export function extractContours(
  grid: number[][],
  xs: number[],
  ys: number[],
  levels: number[]
): ContourLine[] {
  return levels.map((level) => extractContourAtLevel(grid, xs, ys, level));
}
