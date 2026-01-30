import { numpy as np, type Array } from '@jax-js/jax';

export function safeEnergyDiff(
  proposalEnergy: Array,
  initialEnergy: Array
): Array {
  const raw = proposalEnergy.sub(initialEnergy);
  const hasNaN = np.isnan(raw.ref);
  return np.where(hasNaN, np.array(Infinity), raw);
}
