import sys, json, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "../src")
import symbolic
D = "../data/"
def run(args):
    k, t = args
    t0 = time.time()
    try: preds, names = symbolic.solve(t, 20.0)
    except Exception as e: preds, names = [[] for _ in t["test"]], ["ERR " + repr(e)[:80]]
    return k, preds, names, time.time() - t0
for split in ("training", "evaluation"):
    ch = json.load(open(D + f"arc-agi_{split}_challenges.json")); so = json.load(open(D + f"arc-agi_{split}_solutions.json"))
    t0 = time.time()
    with Pool(8) as p: res = p.map(run, ch.items(), chunksize=4)
    att = cor1 = cor2 = 0; wrong = []; progs = {}; slow = max(r[3] for r in res)
    for k, preds, names, dt in res:
        if not any(preds): continue
        att += 1
        ok1 = all(len(pr) and np.array_equal(np.asarray(pr[0]), np.asarray(s)) for pr, s in zip(preds, so[k]))
        ok2 = all(any(np.array_equal(np.asarray(g), np.asarray(s)) for g in pr) for pr, s in zip(preds, so[k]))
        cor1 += ok1; cor2 += ok2
        if not ok2: wrong.append((k, names[:3]))
        for n in names[:1]: progs[n.split("_")[0]] = progs.get(n.split("_")[0], 0) + ok2
    print(f"{split}: tasks={len(ch)} attempted={att} top1={cor1} top2={cor2} precision={cor2/max(1,att):.2f} wall={time.time()-t0:.0f}s slowest={slow:.1f}s")
    print("  correct by family:", progs)
    print("  wrong examples:", wrong[:12])
