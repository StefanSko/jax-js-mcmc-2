/**
 * Distribution definitions for HMC visualization.
 * Each distribution provides a log-density function and visualization bounds.
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
  // Precision matrix (inverse): [[1.333, -0.667], [-0.667, 1.333]]
  const precisionDiag = 1.333;
  const precisionOffDiag = -0.667;

  return {
    name: '2D Gaussian',
    logdensity: (position: Array): Array => {
      // -0.5 * x^T * Precision * x
      // Use np.take to extract elements from 1D position array
      const x = np.take(position.ref, 0);
      const y = np.take(position, 1);

      // Quadratic form: precisionDiag*(x^2 + y^2) + 2*precisionOffDiag*x*y
      const quadForm = x.ref.mul(x.ref).mul(precisionDiag)
        .add(y.ref.mul(y.ref).mul(precisionDiag))
        .add(x.mul(y).mul(2 * precisionOffDiag));

      return quadForm.mul(-0.5);
    },
    bounds: { xMin: -4, xMax: 4, yMin: -4, yMax: 4 },
    initialPosition: [2, 2],
  };
}

/**
 * Banana-shaped distribution (Rosenbrock-like).
 * Educational distribution showing how HMC handles curved posteriors.
 */
export function createBanana(): Distribution {
  const a = 1.0;  // Banana curvature
  const b = 100.0;  // Banana tightness

  return {
    name: 'Banana',
    logdensity: (position: Array): Array => {
      const x = np.take(position.ref, 0);
      const y = np.take(position, 1);

      // Rosenbrock: -(a-x)^2 - b*(y - x^2)^2
      // Scaled down for better sampling
      const term1 = np.subtract(a, x.ref).pow(2);
      const term2 = y.sub(x.pow(2)).pow(2).mul(b);

      return term1.add(term2).mul(-0.05);  // Scale factor for visualization
    },
    bounds: { xMin: -2, xMax: 3, yMin: -1, yMax: 8 },
    initialPosition: [0, 1],
  };
}

/**
 * Neal's funnel distribution.
 * Advanced distribution with varying scales - challenging for samplers.
 */
export function createFunnel(): Distribution {
  return {
    name: 'Funnel',
    logdensity: (position: Array): Array => {
      const v = np.take(position.ref, 0);  // Log-variance
      const x = np.take(position, 1);  // Conditional normal

      // p(v) = N(0, 3^2)
      // p(x|v) = N(0, exp(v))
      // log p(v,x) = -v^2/(2*9) - v/2 - x^2/(2*exp(v))

      const logPv = v.ref.pow(2).mul(-1 / 18);  // -v^2 / (2*9)
      const logPxGivenV = v.ref.mul(-0.5)  // -v/2 (normalization)
        .sub(x.pow(2).div(np.exp(v).mul(2)));  // -x^2 / (2*exp(v))

      return logPv.add(logPxGivenV);
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
