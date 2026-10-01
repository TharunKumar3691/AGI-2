import sys, os, time, json, shutil
sys.path.insert(0, "../src")
os.environ["ARC_INPUT"] = os.path.abspath("fake_input")
import runner
mode = sys.argv[1]
rd = os.path.abspath(f"run_{mode}"); shutil.rmtree(rd, ignore_errors=True)
t0 = time.time()
if mode == "dev":
    runner.run(t0, rerun=False, dev_keys=["0934a4d8", "aa4ec2a5", "20270e3b"], dev_minutes=4, backend="hf",
               model_path=os.path.abspath("tiny_trained"), out_path=os.path.join(rd, "submission.json"), run_dir=rd)
else:
    runner.run(t0, rerun=True, backend="hf", model_path=os.path.abspath("tiny_trained"),
               out_path=os.path.join(rd, "submission.json"), run_dir=rd, time_limit=7 * 60, margin=60, kill_grace=45)
print("WALL", round(time.time() - t0))
