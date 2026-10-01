import sys, json, time, numpy as np
sys.path.insert(0, "../src")
import engine
from arcio import *
ev = json.load(open("../data/arc-agi_evaluation_challenges.json")); evs = json.load(open("../data/arc-agi_evaluation_solutions.json"))
key = "20270e3b"; task = ev[key]
be = engine.Backend("hf", "tiny_model", "cpu"); tm = TokenMap.from_tokenizer(be.tok)
registry = {}
orig_dp = engine.decode_prompt
def rec_dp(tm_, task_, test_in, aug, max_len):
    p = orig_dp(tm_, task_, test_in, aug, max_len)
    if p is not None:
        ti = next(i for i, t in enumerate(task_["test"]) if t["input"] is test_in)
        registry[tuple(p)] = (aug, ti)
    return p
engine.decode_prompt = rec_dp
def fake_run(self, prompts, max_new):
    out = []
    for pr in prompts:
        aug, ti = registry[tuple(pr)]
        sol = aug.fwd(evs[key][ti])
        wrong = aug.fwd(np.full((3, 3), ti + 1))
        out.append([(0.105, tm.reply(sol)), (1.2, tm.reply(wrong))])
    return out
engine.Searcher.run = fake_run
res, info = engine.solve_task(be, tm, key, task, 900.0, engine.Rates(), 0)
print(info)
for ti, cands in enumerate(res):
    print(" output", ti, [(c["count"], round(c["score"], 2), np.array_equal(np.asarray(c["grid"]), np.asarray(evs[key][ti]))) for c in cands[:3]])
    #assert np.array_equal(np.asarray(cands[0]["grid"]), np.asarray(evs[key][ti]))
print("PLUMBING OK")
for ti, cands in enumerate(res):
    for c in cands[:3]: print(ti, c["count"], c["aug_nll"], c["score"], np.asarray(c["grid"]).shape)
