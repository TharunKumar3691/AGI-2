import sys, json, time, numpy as np
sys.path.insert(0, "../src")
import engine
from arcio import *
tr = json.load(open("../data/arc-agi_training_challenges.json")); trs = json.load(open("../data/arc-agi_training_solutions.json"))
key = "833966f4"; task = tr[key]
engine.LR = 5e-5
be = engine.Backend("hf", "tiny_trained", "cpu"); tm = TokenMap.from_tokenizer(be.tok)
rng = np.random.default_rng(0)
augs = make_augs(len(task["train"]), 16, rng)
samples = [train_sample(tm, task, a, 8192) for a in augs]
steps, loss = engine.ttt(be, samples, time.time() + 600)
print("steps", steps, "loss", loss)
# DFS on 4 equal-length prompts
a4 = [a for a in make_augs(len(task["train"]), 1, rng) if not d8_swaps_axes(a.t)]
prompts = [decode_prompt(tm, task, task["test"][0]["input"], a, 7000) for a in a4]
assert len(set(map(len, prompts))) == 1
s = engine.Searcher(be, tm, time.time() + 300)
found = s.run(prompts, tm.max_reply_len() + 1)
print("dfs calls", s.calls, "beams per row", [len(f) for f in found])
sol = np.asarray(trs[key][0]); worst = 0
for a, pr, beams in zip(a4, prompts, found):
    for nll, toks in beams:
        tf = engine.seq_nll(be, tm, [(pr, toks)])[0]
        worst = max(worst, abs(tf - nll))
        g = tm.decode_grid(toks[:-1])
        print(f"  nll={nll:.4f} teacher={tf:.4f} p={np.exp(-nll):.3f} correct={g is not None and np.array_equal(a.inv(g), sol)}")
print("max |dfs - teacher-forced| =", worst)
# greedy check: greedy completion must be among beams if its prob >= 0.2
import torch
with torch.no_grad():
    ids = list(prompts[0]); tot = 0.0
    for _ in range(40):
        lg = be.model(input_ids=torch.tensor([ids])).logits[0, -1]
        lp = torch.log_softmax(lg.float(), -1); allowed = s.allowed
        t = max(allowed, key=lambda z: lp[z]); tot -= float(lp[t]); ids.append(t)
        if t == tm.eos: break
greedy = ids[len(prompts[0]):]
print("greedy p", round(float(np.exp(-tot)), 3), "in beams:", any(b[1] == greedy for b in found[0]))
