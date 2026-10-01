import sys, os, time, shutil
sys.path.insert(0, "../src")
os.environ["ARC_INPUT"] = os.path.abspath("fake_input"); os.environ["ARC_SIM_RERUN"] = "14"
import runner
rd = os.path.abspath("run_sim"); shutil.rmtree(rd, ignore_errors=True)
runner.run(time.time(), rerun=False, backend="hf", model_path=os.path.abspath("tiny_trained"),
           out_path=os.path.join(rd, "submission.json"), run_dir=rd)
