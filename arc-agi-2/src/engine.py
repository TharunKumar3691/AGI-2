"""GPU worker: per-task test-time training + DFS decoding + augmentation re-scoring.

Run as:  python engine.py --rank R --run-dir DIR
Coordinates with the other workers only through files in DIR (claims, status, results).
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
import traceback

import numpy as np

from arcio import (Aug, TokenMap, as_grid, d8_swaps_axes, decode_prompt, encode_query, grid_key,
                   make_augs, stable_seed, task_tokens, text_format, encode_pairs, train_sample)

MAX_SEQ = 8192
LORA = dict(r=256, lora_alpha=32, lora_dropout=0.0, bias="none", use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head"])
LR = 5e-5
WARMUP = 0.1
MAX_NLL = -math.log(0.2)       # DFS keeps continuations with cumulative prob >= 0.2
DFS_CAP = 540.0                # seconds per DFS call
LONG_PROMPT = 4500             # above this many tokens, batch 2 sequences instead of 4 (L4 memory)
N_SCORE_AUG = 8


def log(rank, msg):
    print(f"[gpu{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ backends
class Backend:
    """Wraps model loading and the train/inference mode switches."""

    def __init__(self, kind, model_path, device, seed=42):
        import torch
        self.torch = torch
        self.kind = kind
        self.device = device
        torch.manual_seed(seed)
        if kind == "unsloth":
            from unsloth import FastLanguageModel
            self.FLM = FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=model_path, full_finetuning=False, load_in_4bit=False, local_files_only=True,
                use_gradient_checkpointing=False, max_seq_length=MAX_SEQ)
            model = FastLanguageModel.get_peft_model(
                model, use_gradient_checkpointing=False, random_state=seed, loftq_config=None, **LORA)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
            tok = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
            cfg = dict(LORA)
            cfg["target_modules"] = [m for m in cfg["target_modules"] if m.endswith("proj")]
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", modules_to_save=["embed_tokens", "lm_head"], **cfg))
        if kind == "unsloth":
            for p in model.parameters():
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
        from peft import get_peft_model_state_dict
        self.init_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        self.model, self.tok = model, tok

    def reset(self):
        from peft import set_peft_model_state_dict
        # peft mutates (renames/deletes keys of) the dict it is given -> always pass a fresh shallow copy;
        # load_state_dict copies the tensor values, so the stored tensors are never modified
        set_peft_model_state_dict(self.model, dict(self.init_state))

    def train_mode(self):
        if self.kind == "unsloth":
            self.model = self.FLM.for_training(self.model)
        self.model.train()

    def infer_mode(self):
        if self.kind == "unsloth":
            self.model = self.FLM.for_inference(self.model)
        self.model.eval()


# ============================================================================ training
def lr_at(step, total):
    warm = int(math.ceil(WARMUP * total))
    if step < warm:
        return LR * step / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return LR * 0.5 * (1.0 + math.cos(math.pi * prog))


def ttt(be, samples, deadline):
    """One epoch of LoRA fine-tuning over `samples` (batch size 1), cosine schedule."""
    torch = be.torch
    be.reset()
    be.train_mode()
    params = [p for p in be.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total, done, losses = len(samples), 0, []
    for step, (ids, labels) in enumerate(samples):
        if time.time() > deadline:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total)
        x = torch.tensor([ids], device=be.device)
        y = torch.tensor([labels], device=be.device)
        out = be.model(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        done += 1
    del opt
    be.infer_mode()
    if be.device != "cpu":
        torch.cuda.empty_cache()
    return done, (float(np.mean(losses[-16:])) if losses else float("nan"))


# ============================================================================ decoding
class Searcher:
    """Batched depth-first search over grid tokens, keeping every completion with p >= 0.2."""

    def __init__(self, be, tm, deadline):
        self.be, self.tm = be, tm
        self.allowed = [int(d) for d in tm.digit] + [tm.nl, tm.eos]
        self.allowed_t = be.torch.tensor(self.allowed, device=be.device)
        self.deadline = deadline
        self.calls = 0

    def _lp(self, logits):
        torch = self.be.torch
        lf = logits.float()
        return (lf.index_select(-1, self.allowed_t) - torch.logsumexp(lf, -1, keepdim=True)).cpu().numpy()

    def run(self, prompts, max_new):
        torch = self.be.torch
        with torch.inference_mode():
            x = torch.tensor(prompts, device=self.be.device)
            out = self.be.model(input_ids=x, use_cache=True, return_dict=True)
            lp = self._lp(out.logits[:, -1])
            found = self._search(lp, [0.0] * len(prompts), x.shape[1], out.past_key_values, max_new)
        return [sorted(f, key=lambda z: z[0]) for f in found]

    def _search(self, lp, base, pos, cache, left):
        torch, tm = self.be.torch, self.tm
        n = lp.shape[0]
        found = [[] for _ in range(n)]
        queues = []
        for i in range(n):
            q = []
            if base[i] is not None:
                for j, t in enumerate(self.allowed):
                    s = base[i] - float(lp[i, j])
                    if s < MAX_NLL:
                        if t == tm.eos:
                            found[i].append((s, [t]))
                        elif left > 1:
                            q.append((s, t))
            q.sort()
            queues.append(q)
        while any(queues) and time.time() < self.deadline:
            toks, scores = [], []
            for q in queues:
                if q:
                    s, t = q.pop(0)
                    toks.append(t)
                    scores.append(s)
                else:
                    toks.append(tm.pad)
                    scores.append(None)
            if hasattr(cache, "crop"):          # HF DynamicCache is mutated in place
                cache.crop(pos)
            out = self.be.model(input_ids=torch.tensor(toks, device=self.be.device).view(n, 1),
                                position_ids=torch.full((n, 1), pos, device=self.be.device),
                                past_key_values=cache, use_cache=True, return_dict=True)
            self.calls += 1
            sub = self._search(self._lp(out.logits[:, -1]), scores, pos + 1, out.past_key_values, left - 1)
            for i in range(n):
                for s, suf in sub[i]:
                    found[i].append((s, [toks[i]] + suf))
        return found


def seq_nll(be, tm, pairs_list, bs=4):
    """Teacher-forced NLL of the reply part. pairs_list: [(prompt_ids, reply_ids)]."""
    torch = be.torch
    res = []
    with torch.inference_mode():
        for b in range(0, len(pairs_list), bs):
            chunk = pairs_list[b:b + bs]
            seqs = [p + r for p, r in chunk]
            L = max(map(len, seqs))
            x = torch.tensor([s + [tm.pad] * (L - len(s)) for s in seqs], device=be.device)
            logits = be.model(input_ids=x, use_cache=False, return_dict=True).logits.float()
            norm = torch.logsumexp(logits, -1)
            for i, (p, r) in enumerate(chunk):
                pos = torch.arange(len(p) - 1, len(p) - 1 + len(r), device=be.device)
                tgt = torch.tensor(r, device=be.device)
                res.append(float(-(logits[i, pos, tgt] - norm[i, pos]).sum()))
    return res


# ============================================================================ planning
class Rates:
    """Online-calibrated throughput model (seconds)."""

    def __init__(self):
        self.train = 0.10 / 128      # s per sample-token
        self.dec_fixed = 3.0         # s per augmentation per output
        self.dec_tok = 0.03          # s per augmentation per output-token

    def upd_train(self, secs, tokens):
        if tokens > 0 and secs > 1:
            self.train = 0.6 * self.train + 0.4 * secs / tokens

    def upd_dec(self, secs, n_aug, out_tok):
        pred = n_aug * (self.dec_fixed + self.dec_tok * out_tok)
        if pred > 0 and secs > 1:
            f = min(3.0, max(0.33, secs / pred)) ** 0.4
            self.dec_fixed *= f
            self.dec_tok *= f


def est_out_tokens(task, test_in):
    tr = task["train"]
    tok = lambda g: len(g) * (len(g[0]) + 1)
    if all(np.shape(p["input"]) == np.shape(p["output"]) for p in tr):
        return tok(test_in)
    return int(np.mean([tok(p["output"]) for p in tr]))


def task_cost(task, rates, n_train=128, n_dec=16):
    T = min(task_tokens(task), MAX_SEQ)
    dec = sum(n_dec * (rates.dec_fixed + rates.dec_tok * est_out_tokens(task, t["input"])) for t in task["test"])
    return rates.train * T * n_train + dec + 25.0


def plan(task, rates, budget):
    """Choose (n_train, n_dec) that fits the budget; extra decode passes use leftover time."""
    for n_train, n_dec in [(128, 16), (96, 16), (64, 16), (64, 8), (48, 8), (32, 8), (16, 8)]:
        if task_cost(task, rates, n_train, n_dec) <= budget:
            return n_train, n_dec
    return 16, 8


# ============================================================================ per-task solve
def solve_task(be, tm, key, task, budget, rates, rank, safe=False):
    t0 = time.time()
    deadline = t0 + budget
    n_train, n_dec = (32, 8) if safe else plan(task, rates, budget)
    if be.device != "cpu":
        be.torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(stable_seed(key, "train"))
    n_pairs = len(task["train"])

    # ---- test-time training
    augs = make_augs(n_pairs, (n_train + 7) // 8, rng)
    rng.shuffle(augs)
    samples = [s for s in (train_sample(tm, task, a, MAX_SEQ) for a in augs[:n_train]) if s is not None]
    est_train = rates.train * sum(len(s[0]) for s in samples)
    train_deadline = t0 + min(0.9 * budget, max(0.65 * budget, 1.3 * est_train))
    ts = time.time()
    steps, loss = ttt(be, samples, train_deadline)
    t_train = time.time() - ts
    rates.upd_train(t_train, sum(len(s[0]) for s in samples[:steps]))

    # ---- decoding: passes of 8 D8 augmentations with a fresh colour permutation each
    max_new = tm.max_reply_len() + 1
    outputs = []
    for ti, test in enumerate(task["test"]):
        outputs.append({"cands": {}, "n_aug": 0, "est_tok": est_out_tokens(task, test["input"])})
    drng = np.random.default_rng(stable_seed(key, "decode"))
    max_passes = 4
    passes = [make_augs(n_pairs, 1, drng) for _ in range(max_passes)]
    batches = []  # (pass, test index, [augs])
    for p in range(max_passes):          # breadth first: every output gets 4 augs before any gets 8
        for swap in (False, True):
            for ti in range(len(task["test"])):
                batches.append((p, ti, [a for a in passes[p] if d8_swaps_axes(a.t) == swap]))
    min_passes = n_dec // 8
    ts = time.time()
    for p, ti, group in batches:
        now = time.time()
        if p >= min_passes:
            # extra passes only while the remaining budget comfortably covers them
            est = len(group) * (rates.dec_fixed + rates.dec_tok * outputs[ti]["est_tok"]) * 1.3
            if now + est > deadline:
                continue
        elif now > deadline + 60:
            break
        test_in = task["test"][ti]["input"]
        prompts = [(a, decode_prompt(tm, task, test_in, a, MAX_SEQ - max_new)) for a in group]
        prompts = [(a, pr) for a, pr in prompts if pr is not None]
        by_len = {}
        for a, pr in prompts:
            by_len.setdefault(len(pr), []).append((a, pr))
        chunks = []
        for L, items in sorted(by_len.items()):
            bs = 4 if L <= LONG_PROMPT else 2
            chunks += [items[i:i + bs] for i in range(0, len(items), bs)]
        for items in chunks:
            search = Searcher(be, tm, min(time.time() + DFS_CAP, max(deadline, time.time() + 90)))
            found = search.run([pr for _, pr in items], max_new)
            outputs[ti]["n_aug"] += len(items)
            for (a, _), beams in zip(items, found):
                for nll, toks in beams:
                    g = tm.decode_grid(toks[:-1])
                    if g is None:
                        continue
                    g = a.inv(g)
                    k = grid_key(g)
                    c = outputs[ti]["cands"].setdefault(k, {"grid": g, "count": 0, "beam": [], "aug_nll": None})
                    c["count"] += 1
                    c["beam"].append(nll)
    t_dec = time.time() - ts

    # ---- re-score each unique candidate under fixed augmentations
    ts = time.time()
    for ti, test in enumerate(task["test"]):
        srng = np.random.default_rng(stable_seed(key, ti, "score"))
        saugs = make_augs(n_pairs, 1, srng)
        for c in sorted(outputs[ti]["cands"].values(), key=lambda c: -c["count"]):
            if time.time() > deadline + 120:
                break
            items = []
            for a in saugs:
                pr = decode_prompt(tm, task, test["input"], a, MAX_SEQ - max_new)
                if pr is not None:
                    items.append((pr, tm.reply(a.fwd(c["grid"]))))
            if items:
                long_ = max(len(p) + len(r) for p, r in items) > LONG_PROMPT
                c["aug_nll"] = float(np.mean(seq_nll(be, tm, items, bs=2 if long_ else 4)))
    t_score = time.time() - ts
    tok_sum = sum(o["n_aug"] * o["est_tok"] for o in outputs)
    rates.upd_dec(t_dec + t_score, max(1, sum(o["n_aug"] for o in outputs)),
                  tok_sum / max(1, sum(o["n_aug"] for o in outputs)))

    result = []
    for o in outputs:
        ranked = rank_candidates(list(o["cands"].values()))
        result.append([{"grid": c["grid"].tolist(), "count": c["count"], "aug_nll": c["aug_nll"],
                        "beam": float(min(c["beam"])), "score": c["score"]} for c in ranked[:8]])
    info = dict(budget=round(budget), n_train=n_train, steps=steps, loss=round(loss, 5), t_train=round(t_train),
                t_dec=round(t_dec), t_score=round(t_score), total=round(time.time() - t0),
                n_aug=[o["n_aug"] for o in outputs], n_cand=[len(o["cands"]) for o in outputs],
                mem_gb=round(be.torch.cuda.max_memory_allocated() / 2**30, 1) if be.device != "cpu" else 0)
    return result, info


def rank_candidates(cands):
    for c in cands:
        nll = c["aug_nll"] if c["aug_nll"] is not None else 50.0
        c["score"] = c["count"] - nll
    return sorted(cands, key=lambda c: -c["score"])


# ============================================================================ scheduling (file based)
class Board:
    def __init__(self, run_dir, rank):
        self.dir, self.rank = run_dir, rank
        for sub in ("claims", "status", "results"):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        with open(os.path.join(run_dir, "schedule.json")) as f:
            self.sched = json.load(f)

    def attempts(self, key):
        try:
            with open(os.path.join(self.dir, "attempts", key)) as f:
                return int(f.read() or 0)
        except Exception:
            return 0

    def release(self, key):
        """Put a failed task back in the queue (at most one retry)."""
        os.makedirs(os.path.join(self.dir, "attempts"), exist_ok=True)
        n = self.attempts(key) + 1
        with open(os.path.join(self.dir, "attempts", key), "w") as f:
            f.write(str(n))
        if n < 2:
            try:
                os.remove(os.path.join(self.dir, "claims", key))
            except FileNotFoundError:
                pass

    def claim_next(self):
        for key in self.sched["order"]:
            try:
                fd = os.open(os.path.join(self.dir, "claims", key), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(self.rank).encode())
                os.close(fd)
                return key
            except FileExistsError:
                continue
        return None

    def unclaimed_cost(self):
        claimed = set(os.listdir(os.path.join(self.dir, "claims")))
        return sum(c for k, c in self.sched["cost"].items() if k not in claimed)

    def busy_until(self):
        out = {}
        for f in os.listdir(os.path.join(self.dir, "status")):
            try:
                with open(os.path.join(self.dir, "status", f)) as fh:
                    out[f] = json.load(fh).get("busy_until", 0)
            except Exception:
                pass
        return out

    def set_status(self, **kw):
        path = os.path.join(self.dir, "status", f"gpu{self.rank}")
        with open(path + ".tmp", "w") as f:
            json.dump(kw, f)
        os.replace(path + ".tmp", path)

    def budget(self, key, t_end, n_workers):
        now = time.time()
        busy = self.busy_until()
        others = [max(0.0, t_end - max(now, b)) for w, b in busy.items() if w != f"gpu{self.rank}"]
        others += [t_end - now] * max(0, n_workers - 1 - len(others))
        gpu_left = (t_end - now) + sum(others)
        share = self.sched["cost"][key] / max(1e-9, self.unclaimed_cost() + self.sched["cost"][key])
        b = 0.95 * gpu_left * share
        return max(30.0, min(b, t_end - now - 20, 2.2 * self.sched["cost"][key], 2400))


def cleanup(be):
    import gc
    gc.collect()
    if be.device != "cpu":
        be.torch.cuda.empty_cache()
    try:
        be.infer_mode()
    except Exception:
        pass


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    rank = args.rank
    board = Board(args.run_dir, rank)
    t_end = cfg["t_end"]

    # serialize backend import/initialisation across workers (unsloth patches files on import)
    if rank > 0:
        prev = os.path.join(args.run_dir, "status", f"ready{rank - 1}")
        t_wait = time.time()
        while not os.path.exists(prev) and time.time() < min(t_end, t_wait + 900):
            time.sleep(3)
    t_load = time.time()
    device = "cuda" if cfg["backend"] == "unsloth" else "cpu"
    try:
        be = Backend(cfg["backend"], cfg["model_path"], device)
    finally:   # never block the next worker, even if this one failed to load
        open(os.path.join(args.run_dir, "status", f"ready{rank}"), "w").close()
    tm = TokenMap.from_tokenizer(be.tok)
    log(rank, f"model ready in {time.time() - t_load:.0f}s; tokens digit={tm.digit.tolist()} nl={tm.nl} "
              f"eos={tm.eos} pad={tm.pad} user={tm.user_prefix} mid={tm.mid}")

    with open(cfg["tasks_path"]) as f:
        tasks = json.load(f)
    # verify direct token encoding against the tokenizer on one real task
    k0 = sorted(tasks)[0]
    ref = be.tok.encode(text_format(tasks[k0]["train"][:1], tasks[k0]["test"][0]["input"]), add_special_tokens=False)
    mine = encode_pairs(tm, tasks[k0]["train"][:1], with_labels=False) + encode_query(tm, tasks[k0]["test"][0]["input"])
    if ref != mine:
        log(rank, f"WARNING token encoding mismatch on {k0}: {ref[:24]} vs {mine[:24]}")

    rates = Rates()
    n_done, n_fail_row = 0, 0
    while time.time() < t_end - 60:
        key = board.claim_next()
        if key is None:
            break
        budget = board.budget(key, t_end, cfg["n_workers"])
        board.set_status(busy_until=time.time() + budget, task=key)
        try:
            try:
                result, info = solve_task(be, tm, key, tasks[key], budget, rates, rank)
            except Exception:
                log(rank, f"{key} first attempt failed, retrying in safe mode\n{traceback.format_exc()[-1500:]}")
                cleanup(be)
                left = min(budget, t_end - time.time() - 30)
                if left < 60:
                    raise
                result, info = solve_task(be, tm, key, tasks[key], left, rates, rank, safe=True)
                info["safe"] = True
            with open(os.path.join(args.run_dir, "results", key + ".pkl.tmp"), "wb") as f:
                pickle.dump({"key": key, "outputs": result, "info": info}, f)
            os.replace(os.path.join(args.run_dir, "results", key + ".pkl.tmp"),
                       os.path.join(args.run_dir, "results", key + ".pkl"))
            n_done += 1
            n_fail_row = 0
            log(rank, f"{key} {info}")
        except Exception:
            log(rank, f"{key} FAILED\n{traceback.format_exc()[-1500:]}")
            board.release(key)
            board.set_status(busy_until=0, task=None)
            n_fail_row += 1
            if n_fail_row >= 3:      # something is broken in this process: exit non-zero -> main respawns a fresh one
                log(rank, "3 consecutive failures, exiting for a fresh restart")
                sys.exit(3)
            cleanup(be)
        board.set_status(busy_until=0, task=None)
    log(rank, f"finished {n_done} tasks")


if __name__ == "__main__":
    main()
