import { grad, type Array } from '@jax-js/jax';
import type { Integrator, IntegratorState } from './types';

export function createVelocityVerlet(
  logdensityFn: (position: Array) => Array,
  kineticEnergyFn: (momentum: Array) => Array,
  logdensityAndGradFn?: (position: Array) => [Array, Array]
): Integrator {
  const kineticEnergyGradFn = grad(kineticEnergyFn);
  const logdensityGradFn = logdensityAndGradFn
    ? null
    : grad(logdensityFn);

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

    let newLogdensity: Array;
    let newLogdensityGrad: Array;
    let positionForState = newPosition;

    if (logdensityAndGradFn) {
      positionForState = newPosition.ref;
      [newLogdensity, newLogdensityGrad] = logdensityAndGradFn(newPosition);
    } else {
      newLogdensity = logdensityFn(newPosition.ref);
      if (!logdensityGradFn) {
        throw new Error('logdensityGradFn not initialized');
      }
      newLogdensityGrad = logdensityGradFn(newPosition.ref);
    }

    const newMomentum = momentumHalf.add(newLogdensityGrad.ref.mul(halfStep));

    state.logdensity.dispose();
    state.logdensityGrad.dispose();

    return {
      position: positionForState,
      momentum: newMomentum,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };
  };
}
