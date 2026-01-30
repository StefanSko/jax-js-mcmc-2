/**
 * Distribution definitions for HMC visualization.
 * Each distribution provides a log-density function and visualization bounds.
 *
 * NOTE: JAX-style - use whole-array operations, not element indexing.
 */

import { numpy as np, type Array } from '@jax-js/jax';

export interface TrueParams {
  /** The true mean of the distribution (if well-defined) */
  mean?: [number, number];
  /** The mode(s) of the distribution - where density is highest */
  modes: [number, number][];
  /** Description for the UI */
  description: string;
}

export interface Distribution {
  name: string;
  logdensity: (position: Array) => Array;
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
  initialPosition: [number, number];
  trueParams: TrueParams;
}

/**
 * Standard 2D Gaussian (isotropic).
 * Simple distribution for demonstrating basic HMC.
 */
export function createGaussian2D(): Distribution {
  return {
    name: '2D Gaussian',
    logdensity: (position: Array): Array => {
      // log p(x) = -0.5 * ||x||^2
      return position.ref.mul(position).sum().mul(-0.5);
    },
    bounds: { xMin: -4, xMax: 4, yMin: -4, yMax: 4 },
    initialPosition: [2, 2],
    trueParams: {
      mean: [0, 0],
      modes: [[0, 0]],
      description: 'μ = (0, 0), σ = 1',
    },
  };
}

/**
 * Correlated 2D Gaussian.
 * Demonstrates how HMC handles correlation.
 */
export function createCorrelatedGaussian(): Distribution {
  // Covariance: [[1, 0.9], [0.9, 1]]
  // Precision (inverse): [[5.26, -4.74], [-4.74, 5.26]] (approximately)
  const rho = 0.9;
  const factor = 1 / (1 - rho * rho);

  return {
    name: 'Correlated Gaussian',
    logdensity: (position: Array): Array => {
      // For correlated Gaussian: -0.5 * x^T * Precision * x
      // With precision = [[a, b], [b, a]] where a = 1/(1-rho^2), b = -rho/(1-rho^2)
      // Quadratic form: a*(x^2 + y^2) + 2*b*x*y
      const sumSq = position.ref.mul(position.ref).sum();  // Keep position alive
      // Extract x*y term: (x+y)^2 - x^2 - y^2 = 2xy
      const sumPos = position.sum();  // Consume position here
      const xy2 = sumPos.ref.mul(sumPos).sub(sumSq.ref);  // ref sumPos for double use
      const quadForm = sumSq.mul(factor).sub(xy2.mul(rho * factor * 0.5));
      return quadForm.mul(-0.5);
    },
    bounds: { xMin: -4, xMax: 4, yMin: -4, yMax: 4 },
    initialPosition: [2, -2],
    trueParams: {
      mean: [0, 0],
      modes: [[0, 0]],
      description: 'μ = (0, 0), ρ = 0.9',
    },
  };
}

/**
 * Banana-shaped distribution (Rosenbrock-like).
 * Classic challenging distribution for MCMC.
 */
export function createBanana(): Distribution {
  const a = 1.0;
  const b = 100.0;
  const scale = 0.05; // Scale factor to make it visible

  return {
    name: 'Banana',
    logdensity: (position: Array): Array => {
      // Rosenbrock: f(x,y) = (a-x)^2 + b*(y-x^2)^2
      // We need to extract x and y from the position array
      // Using slicing: x = position[0], y = position[1]
      const x = position.ref.slice([0, 1]).reshape([]);
      const y = position.slice([1, 2]).reshape([]);

      const term1 = np.square(x.ref.mul(-1).add(a)); // (a - x)^2
      const xSq = np.square(x);
      const term2 = np.square(y.sub(xSq)).mul(b); // b * (y - x^2)^2

      return term1.add(term2).mul(-scale);
    },
    bounds: { xMin: -2, xMax: 3, yMin: -1, yMax: 8 },
    initialPosition: [0, 1],
    trueParams: {
      mean: [1, 1], // Approximate - actual mean is hard to compute
      modes: [[1, 1]], // Mode is at (a, a^2) = (1, 1)
      description: 'Mode at (1, 1)',
    },
  };
}

/**
 * Bimodal Gaussian mixture.
 * Two well-separated modes - challenging for HMC without tempering.
 */
export function createBimodal(): Distribution {
  const mode1 = [-2, 0];
  const mode2 = [2, 0];
  const sigma = 0.7;
  const logNorm = -Math.log(2 * Math.PI * sigma * sigma);

  return {
    name: 'Bimodal',
    logdensity: (position: Array): Array => {
      // log(0.5 * N(mode1) + 0.5 * N(mode2))
      // = log(exp(logp1) + exp(logp2)) - log(2)
      const x = position.ref.slice([0, 1]).reshape([]);
      const y = position.slice([1, 2]).reshape([]);

      // Distance squared to each mode
      const d1Sq = np.square(x.ref.sub(mode1[0])).add(np.square(y.ref.sub(mode1[1])));
      const d2Sq = np.square(x.sub(mode2[0])).add(np.square(y.sub(mode2[1])));

      // Log densities (unnormalized, but shifted for numerical stability)
      const logp1 = d1Sq.mul(-0.5 / (sigma * sigma));
      const logp2 = d2Sq.mul(-0.5 / (sigma * sigma));

      // log-sum-exp trick: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
      const maxLog = np.maximum(logp1.ref, logp2.ref);
      const sumExp = np.exp(logp1.sub(maxLog.ref)).add(np.exp(logp2.sub(maxLog.ref)));
      return maxLog.add(np.log(sumExp));
    },
    bounds: { xMin: -5, xMax: 5, yMin: -4, yMax: 4 },
    initialPosition: [-2, 0],
    trueParams: {
      mean: [0, 0], // Symmetric mixture
      modes: [
        [mode1[0], mode1[1]],
        [mode2[0], mode2[1]],
      ],
      description: 'Modes at (-2, 0) and (2, 0)',
    },
  };
}

/**
 * Donut/Ring distribution.
 * Density concentrated on a ring - tests HMC's ability to follow curved paths.
 */
export function createDonut(): Distribution {
  const radius = 2.5;
  const width = 0.4; // Ring width (std dev)

  return {
    name: 'Donut',
    logdensity: (position: Array): Array => {
      // log p(r) ∝ -0.5 * ((r - radius) / width)^2
      const rSq = position.ref.mul(position).sum();
      const r = np.sqrt(rSq);
      const deviation = r.sub(radius);
      return np.square(deviation).mul(-0.5 / (width * width));
    },
    bounds: { xMin: -5, xMax: 5, yMin: -5, yMax: 5 },
    initialPosition: [2.5, 0],
    trueParams: {
      mean: [0, 0], // Center of symmetry
      modes: [[radius, 0], [-radius, 0], [0, radius], [0, -radius]], // Representative points on ring
      description: `Ring at r = ${radius}`,
    },
  };
}

/**
 * Neal's Funnel distribution.
 * Challenging due to varying scale - narrow at top, wide at bottom.
 */
export function createFunnel(): Distribution {
  const vStd = 3;
  const vVar = vStd * vStd;
  const vMode = -0.5 * vVar;

  return {
    name: 'Funnel',
    logdensity: (position: Array): Array => {
      // v ~ N(0, 9), x | v ~ N(0, exp(v))
      // log p(v, x) = -v^2/18 - v/2 - x^2/(2*exp(v))
      const v = position.ref.slice([0, 1]).reshape([]);
      const x = position.slice([1, 2]).reshape([]);

      const logPv = np.square(v.ref).mul(-0.5 / vVar); // -v^2/(2*vVar)
      const expNegV = np.exp(v.ref.mul(-1));
      const logPxGivenV = v.mul(-0.5).sub(np.square(x).mul(expNegV).mul(0.5));

      return logPv.add(logPxGivenV);
    },
    bounds: { xMin: -8, xMax: 8, yMin: -20, yMax: 20 },
    initialPosition: [vMode, 0],
    trueParams: {
      mean: [0, 0],
      modes: [[vMode, 0]],
      description: `v ~ N(0,${vVar}), x|v ~ N(0,eᵛ)`,
    },
  };
}

/**
 * Squiggle - sinusoidal ridge.
 * Tests ability to follow a winding path.
 */
export function createSquiggle(): Distribution {
  const amplitude = 1.5;
  const frequency = 1.0;
  const width = 0.3;

  return {
    name: 'Squiggle',
    logdensity: (position: Array): Array => {
      // Ridge along y = amplitude * sin(frequency * x)
      const x = position.ref.slice([0, 1]).reshape([]);
      const y = position.slice([1, 2]).reshape([]);

      const ridge = np.sin(x.mul(frequency)).mul(amplitude);  // Consume x here
      const deviation = y.sub(ridge);
      return np.square(deviation).mul(-0.5 / (width * width));
    },
    bounds: { xMin: -6, xMax: 6, yMin: -4, yMax: 4 },
    initialPosition: [0, 0],
    trueParams: {
      mean: [0, 0], // Approximate center
      modes: [[0, 0], [Math.PI, 0], [-Math.PI, 0]], // Points on the ridge
      description: `Ridge: y = ${amplitude}·sin(x)`,
    },
  };
}

/**
 * Get all available distributions.
 */
export const distributions: Record<string, () => Distribution> = {
  gaussian: createGaussian2D,
  correlated: createCorrelatedGaussian,
  banana: createBanana,
  bimodal: createBimodal,
  donut: createDonut,
  funnel: createFunnel,
  squiggle: createSquiggle,
};
