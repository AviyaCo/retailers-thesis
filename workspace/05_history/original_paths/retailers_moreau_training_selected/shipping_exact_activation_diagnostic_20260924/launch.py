"""One-shot supervisor: persistent log, exit status and full child invocation timing."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
assert Path.cwd() == root, "Launch from the diagnostic directory"
command = ["/data/home/aviyac/ap_env/bin/python", "-B", "diagnostic.py"]
environment = os.environ.copy()
environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0")
record = {"experiment": root.name, "method": "isolated_synthetic_diagnostic",
          "sample_size": None, "radius": None, "seed": 0, "iterations": None,
          "workers": 1, "threads": 1, "hostname": platform.node(),
          "command": command, "cwd": str(root), "tmux_session": "shipping_exact_activation_20260924",
          "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "status": "RUNNING", "log": str(root / "diagnostic.log"),
          "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / "diagnostic.py", root / "launch.py", root / "README.md")}}
(root / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
start = time.perf_counter()
try:
    with (root / "diagnostic.log").open("x") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   check=False)
    record["exit_code"] = completed.returncode
    record["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
finally:
    record["full_child_invocation_wall_seconds"] = time.perf_counter() - start
    record["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if record["status"] == "RUNNING":
        record["status"] = "FAILED_TO_INVOKE"
    (root / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
(root / "diagnostic.exit").write_text(str(completed.returncode) + "\n")
print(json.dumps(record, indent=2), flush=True)
sys.exit(completed.returncode)