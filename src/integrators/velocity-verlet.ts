import type { Array } from '@jax-js/jax';
import type { Integrator } from './types';

export function createVelocityVerlet(
  _logdensityFn: (position: Array) => Array,
  _kineticEnergyFn: (momentum: Array) => Array
): Integrator {
  throw new Error('Not implemented');
}
