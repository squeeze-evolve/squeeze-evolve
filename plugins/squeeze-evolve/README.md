# squeeze-evolve (plugin)

Adds **`/squeeze-evolve:ask <query>`** — verifier-free evolutionary test-time scaling with
diversity-based multi-model routing. Claude-only, no API keys.

- `commands/ask.md` — the `/squeeze-evolve:ask` entry point (parses args, runs the workflow, presents the result).
- `workflows/squeeze-evolve.js` — the algorithm (deterministic JS run via the Workflow tool).

See the [repository README](../../README.md) for what it does, install steps, flags, and the design notes.

## Quick reference

```
/squeeze-evolve:ask <question>                  # defaults: N=6 K=3 M=2 T=3
/squeeze-evolve:ask <question> --fast           # N=4 K=3 M=2 T=2
/squeeze-evolve:ask <question> --thorough       # N=10 K=4 M=3 T=4
/squeeze-evolve:ask <question> --seed x         # reproducible grouping/routing
```

Routing by group answer-diversity: **consensus → free pick · low → haiku · medium → sonnet · high → opus**.
Init and final synthesis always use the strong model (opus).
