import type { Array } from '@jax-js/jax';
import type { Integrator } from '../integrators/types';

export interface HMCState {
  position: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export interface HMCInfo {
  momentum: Array;
  acceptanceRate: number;
  isAccepted: boolean;
  isDivergent: boolean;
  energy: number;
  numIntegrationSteps: number;
}

export interface HMCConfig {
  stepSize: number;
  numIntegrationSteps: number;
  inverseMassMatrix: Array;
  divergenceThreshold: number;
  integrator: Integrator;
}
