import sys, json, numpy as np, torch
sys.path.insert(0, "../src")
from arcio import *
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("tiny_model"); tm = TokenMap.from_tokenizer(tok)
m = AutoModelForCausalLM.from_pretrained("tiny_model")
tr = json.load(open("../data/arc-agi_training_challenges.json")); trs = json.load(open("../data/arc-agi_training_solutions.json"))
keys = ["833966f4"]
data = []
rng = np.random.default_rng(0)
for k in keys:
    t = dict(tr[k]); full = {"train": t["train"] + [{"input": t["test"][0]["input"], "output": trs[k][0]}]}
    for a in make_augs(len(full["train"]), 8, rng):
        data.append(train_sample(tm, full, a, 4096))
opt = torch.optim.AdamW(m.parameters(), lr=3e-3); m.train()
for ep in range(200):
    tot = 0
    for ids, lab in data:
        out = m(input_ids=torch.tensor([ids]), labels=torch.tensor([lab])); out.loss.backward(); opt.step(); opt.zero_grad(); tot += float(out.loss)
    if ep % 40 == 0: print(ep, tot / len(data))
print("final", tot / len(data))
m.save_pretrained("tiny_trained"); tok.save_pretrained("tiny_trained")
