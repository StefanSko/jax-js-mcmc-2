# jax-js Memory Management Deep Dive

## Overview

**jax-js** is a machine learning library for the web created by Eric Zhang (ekzhang), implementing JAX-style numerical computing in pure JavaScript. It generates WebGPU and WebAssembly kernels for high-performance computation in the browser.

Repository: https://github.com/ekzhang/jax-js

---

## The Core Problem: Memory Management in JavaScript

JavaScript lacks a critical feature that Python ML libraries rely on: **deterministic destructors**. In Python, libraries like PyTorch and JAX use `__del__()` to automatically free GPU/CPU memory when tensors go out of scope. JavaScript has no equivalent mechanism—`FinalizationRegistry` exists but is non-deterministic and cannot be relied upon for timely memory reclamation.

This is a fundamental challenge because:
1. ML workloads use **large arrays** (megabytes to gigabytes of tensor data)
2. Memory must be freed **promptly** to avoid exhausting GPU/WRAM
3. Training loops may create thousands of intermediate tensors

---

## jax-js Solution: Reference-Counted Ownership

jax-js solves this with **Rust-style move semantics** and explicit reference counting. Every `jax.Array` has a reference count that follows these rules:

### The Three Core Rules

| Rule | Effect |
|------|--------|
| **Creation** | When you create an Array, its reference count starts at `1` |
| **Consumption** | Passing an Array to any function decrements refcount by `-1` |
| **Reference** | Accessing `array.ref` returns the same Array and increments refcount by `+1` |

When an Array's reference count reaches `0`, it is **immediately freed** and can no longer be used.

### The `.ref` Pattern

The key insight is that **all functions take ownership** of their arguments. If you want to use an array multiple times, you must explicitly increment its reference count with `.ref`:

```javascript
// BAD: Uses x twice, decrementing its reference count twice
function foo_bad(x, y) {
  return x.add(x.mul(y));  // x is consumed twice!
}

// GOOD: The first usage is x.ref, adding +1 to refcount
function foo_good(x, y) {
  return x.ref.add(x.mul(y));  // x.ref (+1), then x (-1), net = 0
}
```

### Handling Conditional Branches

Every code path must consume each argument exactly once:

```javascript
// BAD: Doesn't consume x in the if-branch
function bar_bad(x, skip) {
  if (skip) return np.zeros(x.shape);  // x is leaked!
  return x;
}

// GOOD: Consumes x once in each branch
function bar_good(x, skip) {
  if (skip) {
    const ret = np.zeros(x.shape);
    x.dispose();  // Explicitly decrement refcount
    return ret;
  }
  return x;
}
```

---

## Backend Memory Architecture

The memory system is divided into **frontend** (jax-js core) and **backend** (device-specific execution).

### Backend Interface

Each backend (CPU, Wasm, WebGPU) implements this interface:

```typescript
interface Backend {
  /** Allocate a new slot with reference count 1 */
  malloc(size: number, initialData?: Uint8Array): Slot;
  
  /** Increment the reference count of the slot */
  incRef(slot: Slot): void;
  
  /** 
   * Decrement the reference count of the slot. 
   * If refcount reaches zero, the slot is freed.
   * Throws if slot was already freed.
   */
  decRef(slot: Slot): void;
  
  /** Read a range of bytes from a buffer */
  read(slot: Slot, start?: number, count?: number): Promise<Uint8Array>;
  
  /** Prepare an expression to be executed later */
  prepare(kernel: Kernel): Promise<Executable>;
  
  /** 
   * Run a prepared operation.
   * Operations run in dispatch order.
   * read() waits for pending operations on that slot.
   */
  dispatch(exe: Executable, inputs: Slot[], outputs: Slot[]): void;
}
```

### Memory Slots

A `Slot` represents a chunk of memory on a device. The backend tracks:
- **Size** in bytes
- **Reference count**  
- **Device-specific handle** (WebGPU buffer, Wasm memory offset, etc.)

---

## How Operations Consume Memory

### Basic Operations

When you perform operations like `x.add(y)`:

1. Both `x` and `y` have their refcounts **decremented**
2. A new output array is allocated with refcount `1`
3. The operation is dispatched to the backend
4. The output is returned

```javascript
const x = np.array([1, 2, 3]);  // x.refcount = 1
const y = np.array([4, 5, 6]);  // y.refcount = 1
const z = x.add(y);             // x.refcount → 0 (freed)
                                // y.refcount → 0 (freed)
                                // z.refcount = 1
```

### Using `.ref` for Multiple Uses

```javascript
const x = np.array([1, 2, 3]);  // x.refcount = 1
const y = x.ref.add(x.ref);     // x.refcount: 1 → 2 → 3 → 2 → 1
// First x.ref: 1 → 2
// Second x.ref: 2 → 3  
// add consumes both: 3 → 2 → 1
// x still alive with refcount 1
```

---

## JIT Compilation and Memory Efficiency

The `jit()` function dramatically improves memory efficiency through **kernel fusion**.

### Without JIT (Eager Execution)

```javascript
// Each operation allocates intermediate buffers
const a = np.sqrt(x);      // Allocates temp1
const b = a.add(2);        // Allocates temp2, frees temp1
const c = b.mul(Math.PI);  // Allocates temp3, frees temp2
const d = c.sum();         // Allocates temp4, frees temp3
```

This requires **4 separate kernel dispatches** and intermediate allocations.

### With JIT (Fused Execution)

```javascript
const f = jit((x) => {
  return np.sqrt(x.add(2).mul(Math.PI)).sum();
});
```

The JIT compiler:
1. **Traces** the computation to build a DAG
2. **Fuses** compatible operations into single kernels
3. **Eliminates** intermediate allocations
4. **Caches** compiled kernels by input shapes

Result: Potentially a **single GPU dispatch** with no intermediate buffers.

---

## WebGPU Memory Considerations

### Buffer Lifecycle

```
1. malloc() → GPUBuffer.createBuffer()
   - Creates GPUBuffer on GPU
   - Sets refcount = 1

2. dispatch() → GPUCommandEncoder + submit()
   - Queues work
   - GPU buffers are "in flight"
   
3. read() → GPUBuffer.mapAsync()
   - Waits for pending work
   - Maps buffer for CPU access
   - Returns data
   
4. decRef() when refcount = 0 → GPUBuffer.destroy()
   - Releases GPU memory
```

### Why Explicit Management Matters for WebGPU

WebGPU has limited VRAM on many devices. Without explicit memory management:
- Buffers would accumulate until GC runs
- GC timing is unpredictable
- Could easily OOM during training

The `.ref` system ensures buffers are freed **immediately** when no longer needed.

---

## Wasm Backend Memory

For the WebAssembly backend, memory is allocated from Wasm linear memory:

```
+------------------------------------------+
|           WASM LINEAR MEMORY             |
+------------------------------------------+
| Header | Slot1 | Slot2 | ... | Free      |
+------------------------------------------+
```

Potential optimizations mentioned in the codebase:
- **Buddy allocator** for tracking free chunks
- Avoiding fragmentation through power-of-2 sizing
- SIMD for 4x throughput on supported operations
- Multi-threading via SharedArrayBuffer

---

## @jax-js/optax: Memory in Optimization

The optax package provides optimizers like Adam and SGD. These maintain **optimizer state** that follows the same ownership rules:

```javascript
import { adam, applyUpdates } from "@jax-js/optax";

let params = np.array([1.0, 2.0, 3.0]);

const solver = adam(1e-3);
let optState = solver.init(params.ref);  // .ref keeps params alive

for (let i = 0; i < 100; i++) {
  const paramsGrad = grad(f)(params.ref);  // .ref keeps params alive
  let updates;
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);  // Old params freed
}
```

Key points:
- `optState` contains momentum/variance buffers (for Adam)
- Each `update()` consumes the old state and returns new state
- Memory is reclaimed each iteration

---

## Comparison: Python vs jax-js Memory Models

| Aspect | Python (JAX/PyTorch) | jax-js |
|--------|---------------------|--------|
| Memory tracking | Automatic via `__del__` | Manual via `.ref` / `.dispose()` |
| Default behavior | Hold reference | Consume (transfer ownership) |
| Multiple uses | Implicit (just use variable) | Explicit (`x.ref`) |
| Cleanup timing | When refcount = 0 via GC | Immediately when refcount = 0 |
| Risk | Accidental retention | Accidental double-free |
| Learning curve | Lower | Steeper, but predictable |

---

## Best Practices

### 1. Always Use `.ref` for Repeated Access

```javascript
// Compute x² + x
function squarePlusX(x) {
  return x.ref.mul(x.ref).add(x);
}
```

### 2. Dispose in All Branches

```javascript
function maybeTransform(x, shouldTransform) {
  if (shouldTransform) {
    return transform(x);  // x consumed by transform
  }
  x.dispose();  // Must consume x in this branch too
  return null;
}
```

### 3. Use JIT for Complex Operations

```javascript
// Bad: Many intermediate allocations
const result = np.exp(x).add(np.exp(y.neg())).log();

// Good: Single fused kernel
const softplus = jit((x, y) => np.exp(x).add(np.exp(y.neg())).log());
const result = softplus(x, y);
```

### 4. Extract Data Before Disposal

```javascript
const x = np.array([1, 2, 3]);
const data = await x.data();  // Get TypedArray
x.dispose();                   // Now safe to dispose
console.log(data);             // data is a copy, still valid
```

---

## Intuition Summary

Think of jax-js arrays like **unique_ptr in C++** or **ownership in Rust**:

1. **Each array has exactly one owner** (the variable holding it)
2. **Passing to a function = transferring ownership**
3. **`.ref` = borrowing** (creates a shared reference)
4. **When owner goes away = memory freed**

This is more work than Python's automatic memory management, but it's:
- **Predictable**: You know exactly when memory is freed
- **Efficient**: No GC pauses, no memory buildup
- **Necessary**: JavaScript has no `__del__` equivalent

The tradeoff is worthwhile for ML workloads where memory efficiency is critical.

---

## Further Reading

- [jax-js GitHub Repository](https://github.com/ekzhang/jax-js)
- [Announcing jax-js Blog Post](https://ss.ekzhang.com/p/jax-js-an-ml-library-for-the-web)
- [JIT Compiler Deep Dive](https://ss.ekzhang.com/p/how-the-jaxjit-jit-compiler-works)
- [JAX Autodidax Tutorial](https://docs.jax.dev/en/latest/autodidax.html)
- [Tinygrad ShapeTracker](https://github.com/tinygrad/tinygrad) (inspiration for jax-js backend)
