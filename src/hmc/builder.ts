import { grad, jit, valueAndGrad as jaxValueAndGrad, type Array } from '@jax-js/jax';
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
  useValueAndGrad?: boolean;
  jitValueAndGrad?: boolean;
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

  valueAndGrad(options: { jit?: boolean } = {}): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, {
      ...this.config,
      useValueAndGrad: true,
      jitValueAndGrad: options.jit ?? false,
    });
  }

  build(): HMCSampler {
    const fullConfig = this.validateAndFillDefaults();
    const metric = createGaussianEuclidean(fullConfig.inverseMassMatrix.ref);
    const logdensityAndGrad = fullConfig.useValueAndGrad
      ? fullConfig.jitValueAndGrad
        ? jit(jaxValueAndGrad(this.logdensityFn))
        : jaxValueAndGrad(this.logdensityFn)
      : null;
    const integrator = createVelocityVerlet(
      this.logdensityFn,
      metric.kineticEnergy,
      logdensityAndGrad ?? undefined
    );
    const step = createHMCKernel(fullConfig, this.logdensityFn, metric, integrator);

    const init = (position: Array): HMCState => {
      if (logdensityAndGrad) {
        const positionForState = position.ref;
        const [logdensity, logdensityGrad] = logdensityAndGrad(position);
        return {
          position: positionForState,
          logdensity,
          logdensityGrad,
        };
      }

      return {
        position: position.ref,
        logdensity: this.logdensityFn(position.ref),
        logdensityGrad: grad(this.logdensityFn)(position),
      };
    };

    return { init, step };
  }

  private validateAndFillDefaults(): HMCConfig & {
    useValueAndGrad: boolean;
    jitValueAndGrad: boolean;
  } {
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
      useValueAndGrad: this.config.useValueAndGrad ?? false,
      jitValueAndGrad: this.config.jitValueAndGrad ?? false,
    };
  }
}

export const HMC = (logdensityFn: (pos: Array) => Array): HMCBuilder => {
  return HMCBuilder.create(logdensityFn);
};
