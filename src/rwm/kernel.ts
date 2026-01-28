import { numpy as np, random, type Array } from '@jax-js/jax';
import type { RWMState, RWMInfo, RWMConfig } from './types';

export type RWMKernel = (key: Array, state: RWMState) => [RWMState, RWMInfo];

export function createRWMKernel(config: RWMConfig): RWMKernel {
  const { logdensityFn, stepSize } = config;

  return function rwmStep(key: Array, state: RWMState): [RWMState, RWMInfo] {
    const [noiseKey, acceptKey] = random.split(key, 2) as unknown as [
      Array,
      Array
    ];

    const noise = random.normal(noiseKey, state.position.shape);
    const proposedPosition = state.position.ref.add(noise.mul(stepSize));

    const proposedLogdensity = logdensityFn(proposedPosition.ref);

    const logRatio = proposedLogdensity.ref.sub(state.logdensity.ref);
    const acceptanceProb = np.minimum(np.array(1.0), np.exp(logRatio));
    const uniform = random.uniform(acceptKey, []);
    const isAccepted = uniform.less(acceptanceProb.ref);

    const proposedPositionInfo = proposedPosition.ref;
    const newPosition = np.where(
      isAccepted.ref,
      proposedPosition.ref,
      state.position.ref
    );
    const newLogdensity = np.where(
      isAccepted.ref,
      proposedLogdensity.ref,
      state.logdensity.ref
    );

    state.position.dispose();
    state.logdensity.dispose();
    proposedPosition.dispose();
    proposedLogdensity.dispose();

    const newState: RWMState = {
      position: newPosition,
      logdensity: newLogdensity,
    };

    const info: RWMInfo = {
      acceptanceProb,
      isAccepted,
      proposedPosition: proposedPositionInfo,
    };

    return [newState, info];
  };
}
