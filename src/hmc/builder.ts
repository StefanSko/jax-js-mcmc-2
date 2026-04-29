import { grad, jit, type Array } from '@jax-js/jax';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import { createHMCKernel } from './kernel';
import { createGaussianEuclidean } from '../metrics/gaussian-euclidean';
import { createVelocityVerlet } from '../integrators/velocity-verlet';

export interface HMCSampler {
  init: (position: Array) => HMCState;
  step: (key: Array, state: HMCState) => [HMCState, HMCInfo];
  dispose: () => void;
}

interface PartialConfig {
  stepSize?: number;
  numIntegrationSteps?: number;
  inverseMassMatrix?: Array;
  divergenceThreshold?: number;
  jitStep?: boolean;
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

  jitStep(value = true): HMCBuilder {
    return new HMCBuilder(this.logdensityFn, {
      ...this.config,
      jitStep: value,
    });
  }

  build(): HMCSampler {
    const fullConfig = this.validateAndFillDefaults();
    const { numIntegrationSteps } = fullConfig;
    const metric = createGaussianEuclidean(fullConfig.inverseMassMatrix);
    const integrator = createVelocityVerlet(
      this.logdensityFn,
      metric.kineticEnergy
    );
    const stepCore = createHMCKernel(
      fullConfig,
      this.logdensityFn,
      metric,
      integrator
    );
    const stepArrays = (
      key: Array,
      state: HMCState
    ): [HMCState, Omit<HMCInfo, 'numIntegrationSteps'>] => {
      const [newState, info] = stepCore(key, state);
      return [
        newState,
        {
          momentum: info.momentum,
          acceptanceProb: info.acceptanceProb,
          isAccepted: info.isAccepted,
          isDivergent: info.isDivergent,
          energy: info.energy,
        },
      ];
    };
    type OwnedFunction<F extends (...args: never[]) => unknown> = F & {
      dispose: () => void;
    };
    type Jit = <Args extends unknown[], R>(
      fn: (...args: Args) => R
    ) => OwnedFunction<(...args: Args) => R>;
    const jitKernel = jit as unknown as Jit;
    let stepJit: OwnedFunction<typeof stepArrays> | undefined;
    const step = fullConfig.jitStep
      ? (() => {
          const compiledStep = jitKernel(stepArrays);
          stepJit = compiledStep;
          return (key: Array, state: HMCState): [HMCState, HMCInfo] => {
            const [newState, info] = compiledStep(key, state);
            return [
              newState,
              { ...info, numIntegrationSteps },
            ];
          };
        })()
      : stepCore;

    const logdensityGradFn = grad(this.logdensityFn);
    const init = (position: Array): HMCState => {
      return {
        position: position.ref,
        logdensity: this.logdensityFn(position.ref),
        logdensityGrad: logdensityGradFn(position),
      };
    };

    let disposed = false;
    const dispose = (): void => {
      if (disposed) {
        return;
      }
      disposed = true;
      stepJit?.dispose();
      metric.dispose();
    };

    return { init, step, dispose };
  }

  private validateAndFillDefaults(): HMCConfig & { jitStep: boolean } {
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
      jitStep: this.config.jitStep ?? true,
    };
  }
}

export const HMC = (logdensityFn: (pos: Array) => Array): HMCBuilder => {
  return HMCBuilder.create(logdensityFn);
};
