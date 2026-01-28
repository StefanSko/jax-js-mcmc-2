import type { Array } from '@jax-js/jax';

export interface RWMState {
  position: Array;
  logdensity: Array;
}

export interface RWMInfo {
  acceptanceProb: Array;
  isAccepted: Array;
  proposedPosition: Array;
}

export interface RWMConfig {
  logdensityFn: (position: Array) => Array;
  stepSize: number;
}
