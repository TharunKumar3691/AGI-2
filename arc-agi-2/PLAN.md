# ARC Prize 2026 – ARC-AGI-2: implementation plan (notebook V1)

## 1. Verified facts (from Kaggle, 2026-09-30)
| Item | Value |
|---|---|
| Task | 240 hidden tasks; predict each test output exactly; 2 attempts per output; score = mean over outputs |
| Submission | `submission.json`, every task id present, `attempt_1` + `attempt_2` for every test input, same order |
| Compute | Notebook only, ≤12 h, **no internet**, L4x4 (4×24 GB), public pre-trained models allowed |
| Daily limit | **1 submission/day** (1 left today) |
| Placeholder test file | 240 tasks copied from *training* (easy, ARC-AGI-1 style) – hidden set is ARC-AGI-2 difficulty |
| Eval set stats | 120 tasks, 172 outputs, prompt median ~1.7k tokens, p90 3.3k, max 7.5k |
| Leaderboard | 1st 83.06 · 3rd 75.42 · 8th (last paid) 36.94 · your best 30.56 |
| Public ceiling | every top public notebook = NVARC 2025 pipeline + Qwen3-4B `sorokin/qwen3_4b_grids15_sft139`; identical copies score **28–34** (run-to-run variance) |
| Timing (from public logs) | TTT ≈ 0.1 s × prompt-tokens per task (128 steps); decode 120–1,080 s per task; 4 sample tasks took 520–1,280 s |
| GPU quota | 15.5 h left until 2026-10-03 (L4x4 counts ×2) → ~7.5 h of test runs |

## 2. Honest assessment
* Top-3 (75–83) must use far stronger private models/data. No public weights reach that; training a new
  4B ARC model is impossible here (no GPU in this container, 7 h quota).
* The realistic ceiling for a from-scratch V1 on public weights is the NVARC class: **~30–36**.
  Goal: beat the public notebooks' *expected* score (~31) by engineering, not luck.
* The public eval set is almost certainly in the model's training data → eval accuracy is only a
  correctness check, never a score estimate.

## 3. Where the public baseline loses points (my levers)
1. **No global time budget.** Tasks run in alphabetical order with a fixed 1,200 s cap; 60 tasks/GPU ×
   ~620 s mean ≈ 10.3 h vs 11.7 h available – a slow draw leaves tail tasks unprocessed = 0 points.
2. **Wasted slack.** Easy tasks finish early; nothing reinvests the saved time.
3. **Non-deterministic candidate scoring** (`hash(str)` seed) → avoidable variance.
4. **Fallback `[[0]]`** for anything unfinished.
5. **Unused CPU** (48 vCPU) while GPUs work.

## 4. Architecture (all code written from scratch)
```
main notebook
 ├─ arcio.py      data, token-level encoding (no string tokenizer on the hot path),
 │                D8 × colour-permutation augmentation + exact inversion, length cutting,
 │                submission build + validation
 ├─ symbolic.py   CPU program search (D8, colour maps, tiling/mirroring, scaling, crops,
 │                object selection, separators+boolean ops, gravity, symmetry repair);
 │                only programs verified on ALL train pairs are kept
 ├─ engine.py     GPU worker: unsloth Qwen3-4B + LoRA r=256, time-bounded custom TTT loop,
 │                batched threshold-DFS decoding (p ≥ 0.2), augmentation-consistency
 │                re-scoring, candidate voting; writes results per task to disk
 └─ runner.py     scheduler: cost model → longest-first queue → per-task budget =
                  remaining GPU-time × task-cost share; watchdog + hard deadline;
                  merges GPU + symbolic results → submission.json
```

### Per-task adaptive budget
`budget_i = 0.92 × (GPU-seconds left) × cost_i / cost_remaining` (capped at 1,500 s).
* budget ≥ full cost → 128 TTT samples + 16 decode augs, then **extra decode augs (up to 32)** while time remains.
* budget < full cost → shrink TTT samples (floor 32) and decode augs (floor 8) so every task is served.
* Costs are re-calibrated online per worker from measured train/decode throughput.

### Decoding and selection
* Prompts batched 4 at a time only within equal-length augmentation classes (needed: the unsloth
  flash-attn patch runs without an attention mask).
* Every unique candidate is re-scored under 8 fixed augmentations (seeded with `crc32`, deterministic).
* Rank = occurrences − mean(augmentation NLL); attempt_1/2 = top-2 distinct grids.

### Merge with symbolic solver
* Model has no candidate → symbolic top-2.
* Symbolic answer is in the model's top-2 → keep the model order.
* Otherwise → attempt_1 = model top-1, attempt_2 = symbolic top-1.

### Robustness
* A valid fallback `submission.json` is written at t = 0 and refreshed as results land.
* Workers stop at `T_end = start + 11 h 25 m`; the watchdog kills stragglers; final assembly + validation.
* Every path found by glob (model, competition data, unsloth utility script) with explicit checks.
* `PYTHONHASHSEED=0`, fixed seeds everywhere.

## 5. Verification before submitting
1. **Local (CPU)**: unit tests for augmentation/inversion round-trips, token encoding vs string format,
   submission schema; symbolic solver precision on training + eval; full engine (TTT, DFS, scoring,
   selection, scheduler, deadline kill) on a tiny random Qwen3 via HF+PEFT.
2. **Kaggle dev kernel** (separate private slug, commit mode, L4x4): 8 eval tasks under a compressed
   deadline → no errors, runtime per phase, budget logic exercised, accuracy vs the baseline's logged
   3/4 on the same tasks.
3. **Final kernel** (new private slug → version 1): short commit run producing a valid `submission.json`,
   then `kaggle competitions submit -k … -v 1`.

## 6. Risks
* L4 commit queue has recently delayed runs by hours.
* The unsloth code path cannot be executed locally – mitigated by mirroring the proven call pattern and
  by the dev kernel run.
* Leaderboard variance of ±3 points means one submission cannot prove an improvement.
