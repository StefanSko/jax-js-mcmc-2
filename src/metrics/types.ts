import type { Array } from '@jax-js/jax';

export interface Metric {
  sampleMomentum: (key: Array, position: Array) => Array;
  kineticEnergy: (momentum: Array) => Array;
  kineticEnergyGrad: (momentum: Array) => Array;
}
