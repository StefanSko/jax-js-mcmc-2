import { numpy as np, random, type Array } from '@jax-js/jax';
import type { HMCState, HMCInfo, HMCConfig } from './types';
import type { Metric } from '../metrics/types';
import type { Integrator, IntegratorState } from '../integrators/types';

export type HMCKernel = (key: Array, state: HMCState) => [HMCState, HMCInfo];

export function createHMCKernel(
  config: HMCConfig,
  _logdensityFn: (position: Array) => Array,
  metric: Metric,
  integrator: Integrator
): HMCKernel {
  const { stepSize, numIntegrationSteps, divergenceThreshold } = config;

  return function hmcStep(key: Array, state: HMCState): [HMCState, HMCInfo] {
    const [keyMomentum, keyAccept] = random.split(key, 2) as unknown as [Array, Array];

    // Sample momentum
    const momentum = metric.sampleMomentum(keyMomentum, state.position.ref);

    // Initial integrator state
    let integState: IntegratorState = {
      position: state.position.ref,
      momentum: momentum.ref,
      logdensity: state.logdensity.ref,
      logdensityGrad: state.logdensityGrad.ref,
    };

    // Compute initial energy: H = -log π(q) + K(p)
    const initialKinetic = metric.kineticEnergy(momentum.ref);
    const initialEnergy = state.logdensity.ref.neg().add(initialKinetic);

    // Integrate trajectory
    for (let i = 0; i < numIntegrationSteps; i++) {
      integState = integrator(integState, stepSize);
    }

    // Compute proposal energy
    const proposalKinetic = metric.kineticEnergy(integState.momentum.ref);
    const proposalEnergy = integState.logdensity.ref.neg().add(proposalKinetic);

    // Metropolis-Hastings acceptance
    const deltaEnergy = proposalEnergy.ref.sub(initialEnergy);
    const isDivergent = np.greater(deltaEnergy.ref, divergenceThreshold);
    const acceptanceProb = np.minimum(np.array(1.0), np.exp(deltaEnergy.neg()));
    const uniform = random.uniform(keyAccept, []);
    const isAccepted = np.less(uniform, acceptanceProb.ref);

    // Select state based on acceptance (branchless with np.where)
    const newPosition = np.where(
      isAccepted.ref,
      integState.position.ref,
      state.position.ref
    );
    const newLogdensity = np.where(
      isAccepted.ref,
      integState.logdensity.ref,
      state.logdensity.ref
    );
    const newLogdensityGrad = np.where(
      isAccepted.ref,
      integState.logdensityGrad.ref,
      state.logdensityGrad.ref
    );

    // Dispose input state arrays (ownership transferred to newState via np.where)
    state.position.dispose();
    state.logdensity.dispose();
    state.logdensityGrad.dispose();
    integState.position.dispose();
    integState.momentum.dispose();
    integState.logdensity.dispose();
    integState.logdensityGrad.dispose();

    const newState: HMCState = {
      position: newPosition,
      logdensity: newLogdensity,
      logdensityGrad: newLogdensityGrad,
    };

    const info: HMCInfo = {
      momentum,
      acceptanceProb,
      isAccepted,
      isDivergent,
      energy: proposalEnergy,
      numIntegrationSteps,
    };

    return [newState, info];
  };
}
