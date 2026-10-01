# Session handoff (2026-09-30 → 2026-10-01)

Paste or point a new conversation at this file. It records everything done, learned and decided in the previous session.

## 0. Who / preferences
- Kaggle user: `tharunkumar369` (GitHub: TharunKumar3691). Wants a brutally honest research partner; always use Kaggle tooling for competitions.
- Never print/commit tokens. No Kaggle write operations (submit, push) unless asked.

## 1. Repo & environment setup (done)
- `AGI-2` repo: `main` created from an empty commit; PR #1 merged → `.claude/hooks/session-start.sh` (pip installs `kaggle` in web sessions; writes `~/.kaggle/kaggle.json` from `KAGGLE_USERNAME`/`KAGGLE_KEY`) + `.claude/settings.json` + `.gitignore` (`.kaggle/`, `kaggle.json`, `.env*`).
- **Kaggle auth in cloud sessions**: no secrets are configured. Two options:
  - A (persistent): add `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME`+`KAGGLE_KEY`) in the cloud environment settings, start a new session.
  - B (per session, what was used): OAuth via FIFO:
    ```sh
    mkfifo /tmp/kfifo
    (nohup sh -c 'sleep 1800 > /tmp/kfifo & kaggle auth login --no-launch-browser < /tmp/kfifo' > /tmp/kaggle_login.log 2>&1 &)
    cat /tmp/kaggle_login.log          # user opens URL, returns code
    echo '<code>' > /tmp/kfifo
    ```
- Reference repo `TharunKumar3691/one-repo` has `CLAUDE.md` + `kaggle-account-and-api-setup.md` (copying them into AGI-2 was blocked by a permission check; user can copy manually).
- Kaggle CLI 2.2.4 tips: `kaggle config set -n competition -v <slug>` then `kaggle competitions pages list --content --page-name rules`; `kaggle competitions topics list|show <id>`; `kaggle competitions submission-limits`; `kaggle kernels logs <ref>`; `kaggle quota`.

## 2. Titanic (done)
- Notebook `tharunkumar369/titanic-fe-soft-voting-ensemble` v2 (private). FE + LogReg/RF/GB/HGB soft vote, CV 0.837 → public LB **0.76315** (prior 0.76076). Data path on Kaggle: `/kaggle/input/competitions/titanic`. Code: `titanic/`.

## 3. User's competitions (status 2026-09-30)
- ARC-AGI-2 (`arc-prize-2026-arc-agi-2`): rank ~615/2298, best 30.56 (own older notebook "V20"). Deadline 2026-11-02, entry/merge 2026-10-26.
- ARC-AGI-3: best 3.64, rank ~474/3494; ~44 versions of LLM-agent prompt tweaks flat at 2–3.6; leader Tufa Labs 45.33. Advice given: stop prompt-tweaking; consider search/state-graph agents.
- Also entered ARC paper track (deadline 2026-11-09).

## 4. ARC-AGI-2 facts (verified)
- 240 hidden tasks; 2 attempts per output; `submission.json` must contain every task id with `attempt_1`/`attempt_2`. **1 submission/day.** ≤12 h, no internet, L4x4 (4×22 GB usable), public pre-trained models allowed.
- Placeholder `arc-agi_test_challenges.json` = 240 *training* tasks (easy). Hidden set ≈ eval difficulty (prompt median ~1.7k tokens, p90 3.3k, max 7.5k).
- Prizes: progress top-8 ($75k…$15k); 8th place on LB ≈ 36.8. Top-3 75–83 (private models).
- All strong public notebooks = NVARC 2025 pipeline: model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, utility kernel `sorokin/pip-install-unsloth-flash-patch` (unsloth 2025.9.7, peft 0.17.1, flash-attn 2.8.2, py3.11, torch 2.8), docker `gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`, machine_shape `NvidiaL4`. Identical copies score **28.06–33.89** (variance). Public eval set likely in model training data → eval scores are inflated (correctness check only).
- Model tokenizer: digits 0–9 = ids 0–9, `\n`=10, `user`=11, `assistant`=12, pad=13, `<|im_start|>`=14, `<|im_end|>`=15. Format: `<|im_start|>user\n{grid}<|im_end|><|im_start|>assistant\n{grid}<|im_end|>`.
- Timing on L4: TTT 128 steps ≈ 0.1 s × prompt tokens; decode 60–1500 s per task; peak GPU mem 12.7–14.3 GB.
- unsloth internals: any call with `past_key_values` goes to the fast 1-token path; every full forward clears the paged KV buffer (so DFS batch size may vary). Batched decode has no attention mask → only equal-length prompts per batch.

## 5. My ARC-AGI-2 solution (code in `arc-agi-2/`)
- `src/arcio.py` (encoding, D8×colour augmentation, submission), `src/engine.py` (GPU worker: unsloth LoRA r=256 TTT, cosine LR 5e-5, DFS p≥0.2, 8-aug rescoring, rank = votes − mean aug NLL), `src/symbolic.py` (CPU verified-program search: 92% precision on training, 0/120 on eval), `src/runner.py` (cost model, longest-first queue, per-task budget = GPU time left × cost share, watchdog, respawn, merge). `make_nb.py dev|final` builds the Kaggle notebooks; `notebook_final/` = pushed version. `PLAN.md` = original plan. `tests/` = local CPU tests (tiny Qwen3; need `pip install torch(cpu) transformers==4.56 peft==0.17.1`).
- Kaggle kernels (private): `tharunkumar369/arc-agi-2-adaptive-ttt-solver` (v1, v2), `tharunkumar369/arc2-adaptive-ttt-devlab` (test kernel; v2 has `ARC_SIM_RERUN` rehearsal mode).
- **Results**: V1 = **0.28** (bug), V2 = **28.06** (submission 56743096). Commit dev checks: 3.0/4 on baseline's logged tasks (parity), 5.0/8 on 8 eval tasks.
- **V1 bug**: `set_peft_model_state_dict` in peft 0.17.1 mutates the dict passed in → 2nd task per worker crashed (KeyError embed_tokens). Fix: pass `dict(self.init_state)`. Plus failure re-queue, 3-failure circuit breaker + respawn, long prompts (>4500 tok) batched in 2s.
- **Honest conclusion**: V2 sits at the bottom of the NVARC variance band; no evidence the scheduling/extra-decode changes help. Rerun logs are hidden by Kaggle.

## 6. Open options (not started)
1. Stop iterating (variance > gains). 2. Controlled A/B vs public baseline over several days. 3. Bigger levers: ensemble a second public ARC model, longer TTT on hard tasks.
- GPU quota ~9 h left until 2026-10-03 reset (L4x4 counts ×2). Competition data is re-downloadable with `kaggle competitions download arc-prize-2026-arc-agi-2`.
