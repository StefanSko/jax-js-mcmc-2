import type { Array } from '@jax-js/jax';

export interface IntegratorState {
  position: Array;
  momentum: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export type Integrator = (
  state: IntegratorState,
  stepSize: number
) => IntegratorState;
