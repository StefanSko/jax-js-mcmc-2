import { describe, it, expect } from 'vitest';
import { numpy as np } from '@jax-js/jax';
import type { RWMState, RWMInfo, RWMConfig } from '../../src/rwm/types';
import '../../src/rwm/types';

describe('RWM types', () => {
  const logdensityFn = (q: np.Array): np.Array => {
    return q.ref.mul(q).mul(-0.5).sum();
  };

  it('RWMState has required fields', () => {
    const state: RWMState = {
      position: np.array([0.0]),
      logdensity: np.array(0.0),
    };

    expect(state.position).toBeDefined();
    expect(state.logdensity).toBeDefined();

    state.position.dispose();
    state.logdensity.dispose();
  });

  it('RWMInfo has required fields', () => {
    const info: RWMInfo = {
      acceptanceProb: np.array(0.5),
      isAccepted: np.array(true),
      proposedPosition: np.array([1.0, 2.0]),
    };

    expect(info.acceptanceProb).toBeDefined();
    expect(info.isAccepted).toBeDefined();
    expect(info.proposedPosition).toBeDefined();

    info.acceptanceProb.dispose();
    info.isAccepted.dispose();
    info.proposedPosition.dispose();
  });

  it('RWMConfig has required fields', () => {
    const config: RWMConfig = {
      logdensityFn,
      stepSize: 0.1,
    };

    expect(config.logdensityFn).toBeTypeOf('function');
    expect(config.stepSize).toBeTypeOf('number');
  });
});
