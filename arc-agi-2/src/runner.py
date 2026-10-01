"""Orchestrator: schedule -> 4 GPU workers + CPU symbolic search -> merged, validated submission."""
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from arcio import empty_submission, find_file, score_submission, validate_submission, write_json_atomic
from engine import Rates, task_cost

SRC = os.path.dirname(os.path.abspath(__file__))


def say(msg):
    print(f"[main {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def locate_model():
    hits = sorted(glob.glob("/kaggle/input/**/qwen3_4b_grids15_sft139/**/config.json", recursive=True))
    if not hits:
        hits = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    assert hits, "model not found under /kaggle/input"
    return os.path.dirname(hits[0])


def unsloth_path(env):
    """None if a plain subprocess already finds unsloth (same env the workers get), else its directory."""
    probe = "import importlib.util,sys; s=importlib.util.find_spec('unsloth'); print(s.origin if s else ''); sys.exit(0 if s else 1)"
    r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        say(f"unsloth found at {r.stdout.strip()}")
        return None
    hits = sorted(glob.glob("/kaggle/input/**/unsloth/__init__.py", recursive=True))
    assert hits, "unsloth not importable and not found under /kaggle/input"
    say(f"unsloth added from {hits[0]}")
    return os.path.dirname(os.path.dirname(hits[0]))


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        return max(1, sum(1 for l in out.splitlines() if l.startswith("GPU")))
    except Exception:
        return 1


def _sym_one(item):
    import symbolic
    k, t = item
    try:
        preds, names = symbolic.solve(t, 20.0)
        return k, preds, names
    except Exception:
        return k, [[] for _ in t["test"]], []


def run_symbolic(tasks, procs=6):
    with Pool(procs) as p:
        res = p.map(_sym_one, sorted(tasks.items()), chunksize=2)
    return {k: {"preds": pr, "programs": nm} for k, pr, nm in res if any(pr)}


def load_results(run_dir):
    out = {}
    for f in glob.glob(os.path.join(run_dir, "results", "*.pkl")):
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            out[r["key"]] = r
        except Exception:
            pass
    return out


def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def merge(tasks, results, symb):
    sub = empty_submission(tasks)
    stats = {"model": 0, "symbolic_only": 0, "symbolic_swap": 0, "fallback": 0}
    for k, t in tasks.items():
        for i, test in enumerate(t["test"]):
            model = [c["grid"] for c in results[k]["outputs"][i]] if k in results and i < len(results[k]["outputs"]) else []
            sym = symb.get(k, {}).get("preds", [[]] * len(t["test"]))[i] if k in symb else []
            att = list(model[:2])
            if att:
                stats["model"] += 1
            for s in sym[:1]:
                if not any(same(s, a) for a in att):
                    if len(att) < 2:
                        stats["symbolic_only" if not att else "symbolic_swap"] += 1
                        att.append(s)
                    else:
                        att[1] = s
                        stats["symbolic_swap"] += 1
            for s in sym[1:]:
                if len(att) < 2 and not any(same(s, a) for a in att):
                    att.append(s)
            if not att:
                stats["fallback"] += 1
                att = [test["input"]]
            if len(att) < 2:
                att.append(att[0])
            sub[k][i] = {"attempt_1": [[int(v) for v in r] for r in att[0]],
                         "attempt_2": [[int(v) for v in r] for r in att[1]]}
    return sub, stats


def run(start_time, rerun, dev_keys=None, dev_minutes=40, backend="unsloth", model_path=None, out_path="submission.json",
        run_dir="/kaggle/working/run", time_limit=12 * 3600, margin=30 * 60, kill_grace=8 * 60):
    for sub_dir in ("", "claims", "status", "results", "attempts"):
        os.makedirs(os.path.join(run_dir, sub_dir), exist_ok=True)
    test_file = find_file("arc-agi_test_challenges.json", roots=(os.environ.get("ARC_INPUT", "/kaggle/input"),))
    comp = os.path.dirname(test_file)
    with open(test_file) as f:
        test_tasks = json.load(f)
    say(f"rerun={bool(rerun)} test tasks={len(test_tasks)} data={comp}")

    # 0. a valid submission exists from the very beginning
    write_json_atomic(empty_submission(test_tasks), out_path)

    sim = os.environ.get("ARC_SIM_RERUN")       # commit-time rehearsal of the rerun path on eval tasks
    sim_solutions = None
    if sim:
        with open(os.path.join(comp, "arc-agi_evaluation_challenges.json")) as f:
            test_tasks = json.load(f)
        with open(os.path.join(comp, "arc-agi_evaluation_solutions.json")) as f:
            sim_solutions = json.load(f)
        rerun, time_limit, margin = True, float(sim) * 60, 5 * 60
        start_time = time.time()
        say(f"SIMULATED RERUN on {len(test_tasks)} eval tasks, {sim} min")
    if rerun:
        gpu_tasks, dev_solutions = test_tasks, None
        t_end = start_time + time_limit - margin
    else:
        with open(os.path.join(comp, "arc-agi_evaluation_challenges.json")) as f:
            ev = json.load(f)
        with open(os.path.join(comp, "arc-agi_evaluation_solutions.json")) as f:
            evs = json.load(f)
        gpu_tasks = {k: ev[k] for k in dev_keys}
        dev_solutions = {k: evs[k] for k in dev_keys}
        t_end = time.time() + dev_minutes * 60

    with open(os.path.join(run_dir, "tasks.json"), "w") as f:
        json.dump(gpu_tasks, f)
    rates = Rates()
    cost = {k: task_cost(t, rates) for k, t in gpu_tasks.items()}
    order = sorted(cost, key=lambda k: -cost[k])
    with open(os.path.join(run_dir, "schedule.json"), "w") as f:
        json.dump({"order": order, "cost": cost}, f)
    ng = n_gpus() if backend == "unsloth" else 2
    model_path = model_path or locate_model()
    cfg = dict(t_end=t_end, n_workers=ng, backend=backend, model_path=model_path,
               tasks_path=os.path.join(run_dir, "tasks.json"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    say(f"gpus={ng} model={model_path} gpu tasks={len(gpu_tasks)} est. GPU-s={sum(cost.values()):.0f} "
        f"available GPU-s={(t_end - time.time()) * ng:.0f} end in {(t_end - time.time()) / 3600:.2f}h")

    # 1. launch GPU workers
    env = dict(os.environ, PYTHONHASHSEED="0", UNSLOTH_DISABLE_STATISTICS="1", OMP_NUM_THREADS="12",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas")
    extra = [SRC] + ([unsloth_path(env)] if backend == "unsloth" else [])
    env["PYTHONPATH"] = os.pathsep.join([p for p in extra if p] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    procs, logs = [], []

    def launch(r):
        e = dict(env, CUDA_VISIBLE_DEVICES=str(r)) if backend == "unsloth" else env
        lf = open(os.path.join(run_dir, f"gpu{r}.log"), "a")
        logs.append(lf)
        return subprocess.Popen([sys.executable, "-u", os.path.join(SRC, "engine.py"), "--rank", str(r),
                                 "--run-dir", run_dir], env=e, stdout=lf, stderr=subprocess.STDOUT, cwd=SRC)

    for r in range(ng):
        procs.append(launch(r))
    restarts = [0] * ng

    # 2. symbolic search on CPU while the models load
    ts = time.time()
    symb = run_symbolic(test_tasks)
    symb_dev = run_symbolic(gpu_tasks) if not rerun else symb
    say(f"symbolic: {len(symb)}/{len(test_tasks)} test tasks with a verified program ({time.time() - ts:.0f}s)")

    # 3. monitor; refresh the submission as results arrive; hard deadline
    t_kill = t_end + kill_grace
    last_n, last_print = -1, 0
    printed = {r: 0 for r in range(ng)}
    while True:
        alive = [p.poll() is None for p in procs]
        unclaimed = len(gpu_tasks) - len(os.listdir(os.path.join(run_dir, "claims")))
        for r, a in enumerate(alive):   # respawn crashed workers while work and time remain
            if not a and procs[r].returncode != 0 and restarts[r] < 3 and unclaimed > 0 and t_end - time.time() > 1200:
                restarts[r] += 1
                say(f"worker {r} exited with code {procs[r].returncode}; respawn #{restarts[r]}")
                procs[r] = launch(r)
                alive[r] = True
        for r, a in enumerate(alive):   # a dead worker contributes no GPU time to the budgets
            if not a:
                write_json_atomic({"busy_until": 1e18, "dead": True}, os.path.join(run_dir, "status", f"gpu{r}"))
        results = load_results(run_dir)
        if len(results) != last_n and rerun:
            sub, _ = merge(test_tasks, results, symb)
            write_json_atomic(sub, out_path)
            last_n = len(results)
        if time.time() - last_print > 300 or not any(alive):
            say(f"done {len(results)}/{len(gpu_tasks)} alive={sum(alive)} left={(t_end - time.time()) / 60:.0f}min")
            for r in range(ng):
                with open(os.path.join(run_dir, f"gpu{r}.log")) as fh:
                    lines = fh.read().splitlines()
                for line in lines[printed[r]:]:
                    if line.startswith("[gpu") or "Error" in line or "Traceback" in line:
                        print(line[:400], flush=True)
                printed[r] = len(lines)
            last_print = time.time()
        if not any(alive):
            break
        if time.time() > t_kill:
            say("hard deadline: terminating workers")
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(20)
    for lf in logs:
        lf.close()

    # 4. final assembly
    results = load_results(run_dir)
    if rerun:
        sub, stats = merge(test_tasks, results, symb)
        if sim_solutions:
            got, n = score_submission(sub, test_tasks, sim_solutions)
            say(f"SIM SCORE (eval, inflated) {got:.2f}/{n} processed {len(results)}/{len(test_tasks)}")
            infos = [r["info"] for r in results.values()]
            say(f"SIM max mem {max([i.get('mem_gb', 0) for i in infos] or [0])} GB, safe-mode tasks "
                f"{sum(1 for i in infos if i.get('safe'))}")
    else:
        sub, stats = merge(test_tasks, {}, symb)
        dev_sub, dev_stats = merge(gpu_tasks, results, symb_dev)
        model_only, _ = merge(gpu_tasks, results, {})
        got, n = score_submission(dev_sub, gpu_tasks, dev_solutions)
        got_m, _ = score_submission(model_only, gpu_tasks, dev_solutions)
        say(f"DEV (eval subset, inflated: the model has likely seen these tasks): merged {got:.2f}/{n}  "
            f"model-only {got_m:.2f}/{n}  processed {len(results)}/{n}  stats={dev_stats}")
        for k in dev_keys:
            if k in results:
                say(f"  {k} {results[k]['info']}")
    validate_submission(sub, test_tasks)
    write_json_atomic(sub, out_path)
    say(f"submission written: {out_path} tasks={len(sub)} merge={stats} processed={len(results)}/{len(gpu_tasks)} "
        f"elapsed={(time.time() - start_time) / 3600:.2f}h")
    return sub
