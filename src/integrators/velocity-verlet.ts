import { grad, type Array } from '@jax-js/jax';
import type { Integrator, IntegratorState } from './types';

export function createVelocityVerlet(
  logdensityFn: (position: Array) => Array,
  kineticEnergyFn: (momentum: Array) => Array
): Integrator {
  const kineticEnergyGradFn = grad(kineticEnergyFn);

  return function velocityVerletStep(
    state: IntegratorState,
    stepSize: number
  ): IntegratorState {
    const halfStep = stepSize * 0.5;

    const momentumHalf = state.momentum.add(
      state.logdensityGrad.ref.mul(halfStep)
    );

    const kineticGrad = kineticEnergyGradFn(momentumHalf.ref);
    const newPosition = state.position.add(kineticGrad.mul(stepSize));

    const newLogdensity = logdensityFn(newPosition.ref);
    const newLogdensityGrad = grad(logdensityFn)(newPosition.ref);

    const newMomentum = momentumHalf.add(newLogdensityGrad.ref.mul(halfStep));

    state.logdensity.dispose();
    state.logdensityGrad.dispose();

    return {
      position: newPosition,
      momentum: newMomentum,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };
  };
}
