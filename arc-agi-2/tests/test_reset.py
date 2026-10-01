import sys, json, time, numpy as np; sys.path.insert(0, "../src")
import engine, peft
from arcio import *
print("peft", peft.__version__)
be = engine.Backend("hf", "tiny_model", "cpu"); tm = TokenMap.from_tokenizer(be.tok)
tr = json.load(open("../data/arc-agi_training_challenges.json")); task = tr["833966f4"]
engine.LR = 1e-2
s = [train_sample(tm, task, a, 8192) for a in make_augs(len(task["train"]), 2, np.random.default_rng(0))]
for rep in range(3):
    engine.ttt(be, s, time.time() + 60)
    be.reset(); cur = peft.get_peft_model_state_dict(be.model)
    print(f"task {rep}: reset ok, max diff to init {max(float((cur[k] - be.init_state[k]).abs().max()) for k in cur)}, init keys {len(be.init_state)}")
