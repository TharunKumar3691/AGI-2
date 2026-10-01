import json, os, sys
import nbformat as nbf
SRC = "src"
def build(out_dir, slug, title, dev_keys, dev_minutes, sim_minutes=None):
    os.makedirs(out_dir, exist_ok=True)
    md = lambda s: nbf.v4.new_markdown_cell(s); code = lambda s: nbf.v4.new_code_cell(s)
    cells = [md(f"""# {title}
Built from scratch for ARC Prize 2026 – ARC-AGI-2.

* **Model**: public NVARC Qwen3-4B grid model (`sorokin/qwen3_4b_grids15_sft139`), LoRA r=256 test-time training per task.
* **Scheduler**: longest-first queue over 4×L4, per-task budget = remaining GPU time × task-cost share,
  online-calibrated throughput model, hard deadline + watchdog → every hidden task gets processed.
* **Decoding**: batched DFS keeping every completion with p ≥ 0.2 over D8 × colour-permutation augmentations
  (breadth-first over test outputs; extra passes when time allows), deterministic (crc32-seeded) re-scoring under 8
  augmentations, rank = votes − mean augmented NLL.
* **CPU in parallel**: verified program search (only programs exact on every train pair), merged conservatively.
* A valid `submission.json` exists from the first second and is refreshed as tasks finish.""")]
    cells.append(code("""import os, sys, time, subprocess
START = time.time()
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
# transformers must not import TensorFlow in this image
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "tensorflow"], capture_output=True)
print("rerun:", RERUN, "| python", sys.version.split()[0])
!nvidia-smi -L"""))
    for f in ("arcio.py", "symbolic.py", "engine.py", "runner.py"):
        cells.append(code(f"%%writefile {f}\n" + open(os.path.join(SRC, f)).read()))
    sim = f'os.environ["ARC_SIM_RERUN"] = "{sim_minutes}"   # rehearsal of the rerun path on eval tasks\n' if sim_minutes else ""
    cells.append(code(sim + f"""sys.path.insert(0, "/kaggle/working")
import runner
DEV_KEYS = {dev_keys!r}   # commit run only: eval tasks used as a correctness check
sub = runner.run(START, RERUN, dev_keys=DEV_KEYS, dev_minutes={dev_minutes},
                 out_path="/kaggle/working/submission.json", run_dir="/kaggle/working/run")
print("tasks in submission.json:", len(sub))"""))
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, os.path.join(out_dir, f"{slug}.ipynb"))
    meta = {"id": f"tharunkumar369/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
            "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
            "machine_shape": "NvidiaL4"}
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=2)
    print("built", out_dir)
which = sys.argv[1]
LOGGED = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
if which == "dev":
    build("kdev", "arc2-adaptive-ttt-devlab", "ARC2 Adaptive TTT DevLab", LOGGED, 30, sim_minutes=55)
else:
    build("kfinal", "arc-agi-2-adaptive-ttt-solver", "ARC-AGI-2 Adaptive TTT Solver",
          LOGGED + ["135a2760", "20270e3b", "2ba387bc", "3e6067c3"], 42)
