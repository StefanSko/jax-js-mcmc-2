import type { Array } from '@jax-js/jax';

export interface HMCState {
  position: Array;
  logdensity: Array;
  logdensityGrad: Array;
}

export interface HMCInfo {
  momentum: Array;
  acceptanceProb: Array;
  isAccepted: Array;
  isDivergent: Array;
  energy: Array;
  numIntegrationSteps: number;
}

export interface HMCConfig {
  stepSize: number;
  numIntegrationSteps: number;
  inverseMassMatrix: Array;
  divergenceThreshold: number;
}
