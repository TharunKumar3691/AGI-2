import sys, json, time, numpy as np
sys.path.insert(0, "../src")
import engine
from arcio import *
tr = json.load(open("../data/arc-agi_training_challenges.json")); trs = json.load(open("../data/arc-agi_training_solutions.json"))
# pick a small same-shape task
key = min((k for k, t in tr.items() if all(np.shape(p["input"]) == np.shape(p["output"]) for p in t["train"])), key=lambda k: task_tokens(tr[k]))
task = tr[key]; print("task", key, "tokens", task_tokens(task), "test shape", np.shape(task["test"][0]["input"]))
engine.LR = 2e-2
be = engine.Backend("hf", "tiny_model", "cpu")
tm = TokenMap.from_tokenizer(be.tok)
rates = engine.Rates()
# repeat TTT data many times via a big budget; n_train is capped at 128 by plan
t0 = time.time()
res, info = engine.solve_task(be, tm, key, task, 400.0, rates, 0)
print("info", info, "wall", round(time.time() - t0, 1))
for ti, cands in enumerate(res):
    sol = np.asarray(trs[key][ti])
    for c in cands[:3]:
        print(" cand", ti, "count", c["count"], "aug_nll", c["aug_nll"] and round(c["aug_nll"], 3), "score", round(c["score"], 3), "correct", np.array_equal(np.asarray(c["grid"]), sol))
# reset check: adapter back to init
before = {k: v.clone() for k, v in __import__("peft").get_peft_model_state_dict(be.model).items()}
be.reset()
after = __import__("peft").get_peft_model_state_dict(be.model)
diff = max(float((after[k] - be.init_state[k]).abs().max()) for k in after)
print("reset max diff", diff, "trained-vs-init diff", max(float((before[k] - be.init_state[k]).abs().max()) for k in before))
