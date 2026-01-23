import { grad, type Array } from '@jax-js/jax';
import type { Integrator, IntegratorState } from './types';

export function createVelocityVerlet(
  logdensityFn: (position: Array) => Array,
  kineticEnergyFn: (momentum: Array) => Array
): Integrator {
  // IMPORTANT: Cache gradient functions - calling grad() repeatedly leaks memory!
  const kineticEnergyGradFn = grad(kineticEnergyFn);
  const logdensityGradFn = grad(logdensityFn);

  return function velocityVerletStep(
    state: IntegratorState,
    stepSize: number
  ): IntegratorState {
    const halfStep = stepSize * 0.5;

    // Half step momentum: p += (ε/2) * ∇log π(q)
    // state.momentum is consumed by add, state.logdensityGrad.ref keeps it alive
    const momentumHalf = state.momentum.add(
      state.logdensityGrad.ref.mul(halfStep)
    );

    // Full step position: q += ε * ∇K(p)
    // momentumHalf.ref keeps it alive for later use
    const kineticGrad = kineticEnergyGradFn(momentumHalf.ref);
    // state.position is consumed by add
    const newPosition = state.position.add(kineticGrad.mul(stepSize));

    // Compute new logdensity and gradient at new position
    const newLogdensity = logdensityFn(newPosition.ref);
    const newLogdensityGrad = logdensityGradFn(newPosition.ref);

    // Half step momentum: p += (ε/2) * ∇log π(q_new)
    // momentumHalf is consumed by add, newLogdensityGrad.ref keeps it alive
    const newMomentum = momentumHalf.add(newLogdensityGrad.ref.mul(halfStep));

    // Dispose old state arrays that weren't consumed by operations
    // state.position - consumed by add
    // state.momentum - consumed by add
    state.logdensity.dispose();     // not consumed, dispose
    state.logdensityGrad.dispose(); // only used via .ref, dispose

    return {
      position: newPosition,
      momentum: newMomentum,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };
  };
}
