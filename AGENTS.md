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
