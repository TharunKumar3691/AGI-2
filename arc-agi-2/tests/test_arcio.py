import sys, json, numpy as np
sys.path.insert(0, "../src")
from arcio import *
from transformers import PreTrainedTokenizerFast
D = "../data/"
ev = json.load(open(D + "arc-agi_evaluation_challenges.json")); tr = json.load(open(D + "arc-agi_training_challenges.json"))
rng = np.random.default_rng(0)
# 1. round trips
n = 0
for t in list(ev.values()):
    for p in t["train"]:
        for g in (p["input"], p["output"]):
            for a in make_augs(len(t["train"]), 2, rng):
                assert np.array_equal(a.inv(a.fwd(g)), np.asarray(g)); n += 1
print("roundtrips ok", n)
# D8 group has 8 distinct elements on an asymmetric grid, swap flags right
g = np.arange(6).reshape(2, 3)
outs = {d8_fwd(g, t).tobytes() + bytes(d8_fwd(g, t).shape) for t in range(8)}
assert len(outs) == 8
assert all((d8_fwd(g, t).shape == (3, 2)) == d8_swaps_axes(t) for t in range(8))
print("d8 ok")
# 2. token encoding == tokenizer on string format
tok = PreTrainedTokenizerFast.from_pretrained("tiny_model")
tm = TokenMap.from_tokenizer(tok)
for k, t in list(ev.items()) + list(tr.items())[:200]:
    ref = tok.encode(text_format(t["train"], t["test"][0]["input"]), add_special_tokens=False)
    mine = encode_pairs(tm, t["train"], with_labels=False) + encode_query(tm, t["test"][0]["input"])
    assert ref == mine, k
    # grid decode round trip
    for p in t["train"]:
        assert np.array_equal(tm.decode_grid(tm.grid(p["output"])), np.asarray(p["output"]))
print("encoding ok; max reply len", tm.max_reply_len())
# labels only on outputs + eos
ids, lab = encode_pairs(tm, ev[sorted(ev)[0]]["train"][:1])
lab_tokens = [i for i, l in zip(ids, lab) if l != -100]
assert lab_tokens == tm.reply(ev[sorted(ev)[0]]["train"][0]["output"])
print("labels ok")
# 3. fit_pairs drops leading pairs
big = max(ev.values(), key=task_tokens)
a = make_augs(len(big["train"]), 1, rng)[0]
pr = decode_prompt(tm, big, big["test"][0]["input"], a, 8192 - tm.max_reply_len() - 1)
print("biggest task prompt len", len(pr) if pr else None, "tokens est", task_tokens(big))
s = train_sample(tm, big, a, 3000)
print("cut train sample len", len(s[0]) if s else None)
# 4. submission helpers
sub = empty_submission(ev); assert validate_submission(sub, ev)
sol = json.load(open(D + "arc-agi_evaluation_solutions.json"))
print("empty score", score_submission(sub, ev, sol))
