import { jit, type Array } from '@jax-js/jax';
import type { RWMState, RWMInfo, RWMConfig } from './types';
import { createRWMKernel } from './kernel';

export interface RWMSampler {
  init: (position: Array) => RWMState;
  step: (key: Array, state: RWMState) => [RWMState, RWMInfo];
  dispose: () => void;
}

interface PartialConfig {
  stepSize?: number;
  jitStep?: boolean;
}

export class RWMBuilder {
  private constructor(
    private readonly logdensityFn: (pos: Array) => Array,
    private readonly config: PartialConfig
  ) {}

  static create(logdensityFn: (pos: Array) => Array): RWMBuilder {
    return new RWMBuilder(logdensityFn, {});
  }

  stepSize(value: number): RWMBuilder {
    return new RWMBuilder(this.logdensityFn, { ...this.config, stepSize: value });
  }

  jitStep(value = true): RWMBuilder {
    return new RWMBuilder(this.logdensityFn, { ...this.config, jitStep: value });
  }

  build(): RWMSampler {
    const fullConfig = this.validateAndFillDefaults();
    const kernel = createRWMKernel({
      logdensityFn: this.logdensityFn,
      stepSize: fullConfig.stepSize,
    });
    type OwnedFunction<F extends (...args: never[]) => unknown> = F & {
      dispose: () => void;
    };
    type Jit = <Args extends unknown[], R>(
      fn: (...args: Args) => R
    ) => OwnedFunction<(...args: Args) => R>;
    const jitKernel = jit as unknown as Jit;
    const stepJit = fullConfig.jitStep ? jitKernel(kernel) : undefined;
    const step = stepJit ?? kernel;

    const init = (position: Array): RWMState => {
      return {
        position: position.ref,
        logdensity: this.logdensityFn(position),
      };
    };

    let disposed = false;
    const dispose = (): void => {
      if (disposed) {
        return;
      }
      disposed = true;
      stepJit?.dispose();
    };

    return { init, step, dispose };
  }

  private validateAndFillDefaults(): Pick<RWMConfig, 'stepSize'> & { jitStep: boolean } {
    if (this.config.stepSize === undefined) {
      throw new Error('stepSize required');
    }

    return {
      stepSize: this.config.stepSize,
      jitStep: this.config.jitStep ?? true,
    };
  }
}

export const RWM = (logdensityFn: (pos: Array) => Array): RWMBuilder => {
  return RWMBuilder.create(logdensityFn);
};
