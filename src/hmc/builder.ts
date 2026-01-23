import { grad, type Array } from '@jax-js/jax';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import { createHMCKernel } from './kernel';
import { createGaussianEuclidean } from '../metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../integrators/velocity-verlet';

export interface HMCSampler {
  init: (position: Array) => HMCState;
  step: (key: Array, state: HMCState) => [HMCState, HMCInfo];
}

interface PartialConfig {
  stepSize?: number;
  numIntegrationSteps?: number;
  inverseMassMatrix?: Array;
  divergenceThreshold?: number;
}

export class HMCBuilder {
  private constructor(
    private readonly logdensityFn: (pos: Array) => Array,
    private readonly config: PartialConfig
  ) {}

  static create(logdensityFn: (pos: Array) => Array): HMCBuilder {
    return new HMCBuilder(logdensityFn, {});
  }

  stepSize(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, { ...this.config, stepSize: value });
  }

  numIntegrationSteps(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, {
      ...this.config,
      numIntegrationSteps: value,
    });
  }

  inverseMassMatrix(value: Array): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, {
      ...this.config,
      inverseMassMatrix: value,
    });
  }

  divergenceThreshold(value: number): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, {
      ...this.config,
      divergenceThreshold: value,
    });
  }

  build(): HMCSampler {
    const fullConfig = this.validateAndFillDefaults();
    const metric = createGaussianEuclidean(fullConfig.inverseMassMatrix.ref);
    const integrator = createVelocityVerlet(this.logdensityFn, metric.kineticEnergy);
    const step = createHMCKernel(fullConfig, this.logdensityFn, metric, integrator);

    const init = (position: Array): HMCState => {
      return {
        position: position.ref,
        logdensity: this.logdensityFn(position.ref),
        logdensityGrad: grad(this.logdensityFn)(position),
      };
    };

    return { init, step };
  }

  private validateAndFillDefaults(): HMCConfig {
    if (this.config.stepSize === undefined) {
      throw new Error('stepSize required');
    }
    if (this.config.numIntegrationSteps === undefined) {
      throw new Error('numIntegrationSteps required');
    }
    if (this.config.inverseMassMatrix === undefined) {
      throw new Error('inverseMassMatrix required');
    }

    return {
      stepSize: this.config.stepSize,
      numIntegrationSteps: this.config.numIntegrationSteps,
      inverseMassMatrix: this.config.inverseMassMatrix,
      divergenceThreshold: this.config.divergenceThreshold ?? 1000,
    };
  }
}

export const HMC = (logdensityFn: (pos: Array) => Array): HMCBuilder => {
  return HMCBuilder.create(logdensityFn);
};
