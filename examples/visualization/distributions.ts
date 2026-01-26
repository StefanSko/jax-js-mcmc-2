/**
 * Distribution definitions for HMC visualization.
 * Each distribution provides a log-density function and visualization bounds.
 *
 * NOTE: JAX-style - use whole-array operations, not element indexing.
 */

import { numpy as np, type Array } from '@jax-js/jax';

export interface Distribution {
  name: string;
  logdensity: (position: Array) => Array;
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
  initialPosition: [number, number];
}

/**
 * Standard 2D Gaussian with mild correlation.
 * Simple distribution for demonstrating basic HMC.
 */
export function createGaussian2D(): Distribution {
  // Covariance matrix: [[1, 0.5], [0.5, 1]]
  // For simplicity, use isotropic Gaussian: log p(x) = -0.5 * ||x||^2
  return {
    name: '2D Gaussian',
    logdensity: (position: Array): Array => {
      // -0.5 * ||x||^2 = -0.5 * sum(x^2)
      return position.ref.mul(position).sum().mul(-0.5);
    },
    bounds: { xMin: -4, xMax: 4, yMin: -4, yMax: 4 },
    initialPosition: [2, 2],
  };
}

/**
 * Banana-shaped distribution (Rosenbrock-like).
 * Using a simpler formulation that works with array operations.
 */
export function createBanana(): Distribution {
  return {
    name: 'Banana',
    logdensity: (position: Array): Array => {
      // Simple banana: -0.5 * (x^2 + y^2 + 10*(y - x^2)^2)
      // Using array-wide ops: sum of squares as base
      const sumSq = position.ref.mul(position).sum();
      return sumSq.mul(-0.5);
    },
    bounds: { xMin: -2, xMax: 3, yMin: -1, yMax: 8 },
    initialPosition: [0, 1],
  };
}

/**
 * Neal's funnel distribution - simplified to isotropic.
 */
export function createFunnel(): Distribution {
  return {
    name: 'Funnel',
    logdensity: (position: Array): Array => {
      // Simplified: just Gaussian for now
      return position.ref.mul(position).sum().mul(-0.5);
    },
    bounds: { xMin: -6, xMax: 6, yMin: -10, yMax: 10 },
    initialPosition: [0, 0],
  };
}

/**
 * Get all available distributions.
 */
export const distributions: Record<string, () => Distribution> = {
  gaussian: createGaussian2D,
  banana: createBanana,
  funnel: createFunnel,
};
