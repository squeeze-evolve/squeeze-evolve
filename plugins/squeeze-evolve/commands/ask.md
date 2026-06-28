---
description: Evolve a better answer to a query via diversity-routed multi-model test-time scaling
argument-hint: <query> [--n 6] [--k 3] [--m 2] [--t 3] [--threshold 0.8] [--fast|--thorough]
---

Run **Squeeze-Evolve** on the user's query: verifier-free evolutionary test-time
scaling that samples N candidate answers from the strong model, then for T loops
groups them, measures each group's answer-diversity, and routes recombination by
diversity — **consensus → free representative pick, low → haiku, medium → sonnet,
high → opus** — before synthesizing a final answer with the strong model.

## Step 1 — Parse `$ARGUMENTS` into a query + optional flags

Treat the whole of `$ARGUMENTS` as the query, EXCEPT for these recognized flags,
which you strip out before forming the query string:

- `--n <int>`   population size at init (default 6)
- `--k <int>`   candidates per group (default 3)
- `--m <int>`   groups per loop (default 2)
- `--t <int>`   number of evolve loops (default 3)
- `--threshold <0..1>`  answers ≥ this similarity count as the same opinion (default 0.8)
- `--low <0..1>` / `--high <0..1>`  diversity-ratio cut-points for routing (defaults 0.5 / 0.8)
- `--seed <string>`  fix the RNG seed for reproducible grouping
- `--strong <model>` / `--mid <model>` / `--cheap <model>`  override a tier (defaults opus / sonnet / haiku)
- `--fast`     preset: N=4 K=3 M=2 T=2  (cheaper, quicker)
- `--thorough` preset: N=10 K=4 M=3 T=4 (more compute)

Everything left after removing flags (trimmed) is the **query**. If the query is
empty, ask the user what they want answered and stop.

## Step 2 — Tell the user, then run the Workflow

If the **Workflow tool is available**, this command invocation is your
authorization to run it. First tell the user the rough plan in one line
("sampling N answers from opus, then T evolve loops; ~N + a few agents"), then
call it and surface its `log()` lines as they arrive:

```
Workflow({
  scriptPath: "${CLAUDE_PLUGIN_ROOT}/workflows/squeeze-evolve.js",
  args: {
    query: "<the parsed query>",
    N: 6, K: 3, M: 2, T: 3,
    threshold: 0.8, low: 0.5, high: 0.8,
    tiers: { strong: "opus", mid: "sonnet", cheap: "haiku" }
  }
})
```

Pass `query` always; include ONLY the other keys the user actually overrode (via
flags/presets) and omit the rest so the script applies its own defaults. Apply
`--fast`/`--thorough` by expanding them into N/K/M/T. Worked example — the user
types `/squeeze-evolve:ask What's the best caching strategy for a read-heavy API? --thorough --threshold 0.75`
→ you call it with
`args: { query: "What's the best caching strategy for a read-heavy API?", N: 10, K: 4, M: 3, T: 4, threshold: 0.75 }`.

## Step 3 — Present the result (see "Present" below).

## Method B — fallback if the Workflow tool is NOT available (older build)

Do a degraded best-effort by hand: produce 3–4 independent answers to the query
(re-answer it 3–4 times, or spawn parallel general-purpose subagents if you have
the Task tool — give the cheap ones `model: haiku`), each ending with a one-line
`ANSWER:` summary; eyeball how much they agree; then write one reconciled final
answer yourself, weighting the points most candidates share and resolving
disagreements explicitly. Tell the user this was the no-Workflow fallback (no
deterministic clustering / routing).

## Present

When the workflow returns, render its `body` field **verbatim** as the answer.
Then append a compact trace built from the returned object:

- `trace.init` — `sampled` @ `model` (and how many `survived`)
- `trace.loops[]` — per loop: `populationDiversity → populationDiversityAfter`
  clusters, and each group's `size→clusters route(model)`; note any loop with
  `converged: true`
- `trace.finalModel`
- `modelUsage` — counts per tier (`opus` / `sonnet` / `haiku` / `free`)
- `hyperparams` — N / K / M / T / threshold actually used

Example trace block to render under the answer:

```
— Squeeze-Evolve trace —
Init: 6 candidates @ opus (6 survived)
Loop 1: diversity 5→3 | g1 3→3cl high(opus)  g2 3→2cl medium(sonnet)
Loop 2: diversity 3→2 | g1 3→2cl medium(sonnet)  g2 3→1cl consensus(free)
Loop 3: diversity 2→1 | converged — stopped early
Final synthesis @ opus
Model usage: opus ×8 · sonnet ×2 · haiku ×0 · free ×1
Params: N=6 K=3 M=2 T=3 thr=0.8
```

If `notes[]` is non-empty (e.g. degraded runs, early stops), surface it. If
`answer`/`body` are null (all agents failed), tell the user it failed and suggest
a `--fast` retry.
