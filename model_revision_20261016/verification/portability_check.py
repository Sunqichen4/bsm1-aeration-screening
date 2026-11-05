"""Post-freeze relocation audit; does not fit or generate native labels."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="relocation smoke")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("Run name must be one folder name")
    copied = ROOT / ".runtime_cache" / args.run_name / "复制程序"
    if copied.exists():
        raise FileExistsError("Preserve previous relocation evidence; choose a new run directory")
    copied.mkdir(parents=True)
    freeze = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
    for record in freeze["files"]:
        source = ROOT / record["path"]
        target = copied / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for folder in ["legacy", "reference", "reference_trajectories", "numerical"]:
        shutil.copytree(ROOT / folder, copied / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbc", "*.nbi"))
    for name in ["freeze.json", "reference_summary.csv", "reference_provenance.json", "reference_execution.json"]:
        shutil.copy2(ROOT / name, copied / name)
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(copied.parent / "numba_relocation")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        env[key] = "1"
    results = []
    replay = '''import run_study as s
import json,numpy as np,pandas as pd
f=pd.read_csv("new_candidates.csv",float_precision="round_trip")
m=s.read(s.ROOT/"outputs/development/models.json")
p=s.all_predict(f,m,finite=True)[["candidate_id","model","predicted_peak"]]
expected=pd.read_csv("frozen_predictions.csv",float_precision="round_trip")
z=p.merge(expected,on=["candidate_id","model"],validate="one_to_one",suffixes=("_replay","_frozen"))
err=float(np.max(np.abs(z.predicted_peak_replay-z.predicted_peak_frozen)))
raw_errors=int(((z.predicted_peak_replay<=4.)!=z.raw_compliant).sum())
release=(z.predicted_peak_replay+z.buffer_mgN_L<4.)&z.in_joint_domain&z.in_feature_domain
reject=(z.predicted_peak_replay-z.buffer_mgN_L>4.)&z.in_joint_domain&z.in_feature_domain
gate_errors=int(((release!=z.direct_release)|(reject!=z.direct_reject)|((~(release|reject))!=z.recheck)).sum())
print(json.dumps({"prediction_pairs":len(z),"maximum_absolute_difference":err,"raw_decision_differences":raw_errors,"gate_differences":gate_errors,"passed":len(z)==540 and err<1e-9 and raw_errors==0 and gate_errors==0}))
assert len(z)==540 and err<1e-9 and raw_errors==0 and gate_errors==0
'''
    commands = [["fresh_reference.py", ".", "--audit"],
                ["fresh_reference.py", ".", "--audit-half-step"],
                ["-c", replay]]
    for command in commands:
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", *command], cwd=copied,
                                env=env, text=True, encoding="utf-8", capture_output=True)
        results.append({"command": command[:3], "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            break
    report = {"passed": len(results) == 3 and all(x["returncode"] == 0 for x in results),
              "role": "post_freeze_relocation_and_prediction_replay_no_fit_no_new_native_labels",
              "copied_directory": str(copied), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": results}
    (ROOT / "portability_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["passed", "role", "copied_directory"]}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Relocation audit failed; inspect portability_validation.json")

if __name__ == "__main__":
    main()
