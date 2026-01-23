import { grad, numpy as np } from '@jax-js/jax';

const iterations = Number.parseInt(process.env.ITERATIONS ?? '1000', 10);
const logEvery = Number.parseInt(process.env.LOG_EVERY ?? '100', 10);
const stepSize = Number.parseFloat(process.env.STEP_SIZE ?? '0.05');

const f = (x: np.Array): np.Array => x.ref.mul(x).sum();
const gradFn = grad(f);

let params = np.array([1.0, 2.0, 3.0]);

const logMemory = (label: string): void => {
  if (globalThis.gc) {
    globalThis.gc();
  }
  const { heapUsed, rss } = process.memoryUsage();
  const heapMb = (heapUsed / 1024 / 1024).toFixed(1);
  const rssMb = (rss / 1024 / 1024).toFixed(1);
  console.log(`${label} heap=${heapMb}MB rss=${rssMb}MB`);
};

logMemory('start');

for (let i = 0; i < iterations; i++) {
  const grads = gradFn(params.ref);
  const update = grads.mul(stepSize);
  const nextParams = params.sub(update);

  params = nextParams;

  if ((i + 1) % logEvery === 0) {
    logMemory(`iter ${i + 1}`);
  }
}

logMemory('end');

params.dispose();
