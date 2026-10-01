import sys, json, time, numpy as np, torch, math
sys.path.insert(0, "../src")
import engine
from arcio import *
print(engine.lr_at(0, 10), engine.lr_at(1,10), engine.lr_at(5,10), "LR", engine.LR)
be = engine.Backend("hf", "tiny_model", "cpu")
ps = [p for p in be.model.parameters() if p.requires_grad]; print("trainable", len(ps), sum(p.numel() for p in ps))
tr = json.load(open("../data/arc-agi_training_challenges.json")); task = tr["833966f4"]
tm = TokenMap.from_tokenizer(be.tok)
ids, lab = train_sample(tm, task, make_augs(len(task["train"]),1,np.random.default_rng(0))[0], 8192)
be.train_mode()
opt = torch.optim.AdamW(ps, lr=1e-2)
for i in range(30):
    x = torch.tensor([ids]); y = torch.tensor([lab])
    out = be.model(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
    out.loss.backward(); gn = torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step(); opt.zero_grad()
    if i % 5 == 0: print(i, float(out.loss), float(gn))
