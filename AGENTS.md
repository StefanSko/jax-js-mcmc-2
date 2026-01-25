# Agent Instructions

This project uses AI coding assistants. See the relevant file for your agent:

- **Claude Code**: See [CLAUDE.md](./CLAUDE.md)
- **Other agents**: See [CLAUDE.md](./CLAUDE.md) for project context

## Quick Start

1. Read `CLAUDE.md` for memory model and TDD workflow
2. Read `docs/plans/2026-01-23-jax-js-hmc-design.md` for full design
3. Reference repos are in `/tmp/jax-js` and `/tmp/blackjax`

## Key Commands

```bash
npm run ci          # Full check: typecheck + lint + test
npm run test:watch  # TDD mode
npm run typecheck   # Types only
npm run lint        # Lint only
```

## Mandatory Integration Check (Memory)

Run after changes that touch HMC, integrator, or memory management:

```bash
JAXJS_CACHE_LOG=1 NODE_OPTIONS="--expose-gc --loader ./tools/jaxjs-loader.mjs" \
  ITERATIONS=2000 LOG_EVERY=500 npx tsx examples/memory-profile-hmc-jit-step.ts
```

**Desired output:** memory should plateau (heap and rss do not trend upward) and
stay comfortably below ~300MB by the end of 2,000 iterations. If it grows
monotonically or exceeds ~500MB, treat as a regression.

**Why prefer JIT mode:** eager mode runs each primitive immediately and
allocates many intermediates per step, pushing allocator high-water marks.
`jitStep()` fuses the whole step into a compiled kernel, reducing allocations
and stabilizing memory.
