"""Fresh-output restart of the approved matched nominal timing trial."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
IMPLEMENTATION = WORKSPACE / "retailers_moreau_N50_R3_net16_v2"
sys.path.insert(0, str(IMPLEMENTATION))

import run_trial


if __name__ == "__main__":
    # Reuse tested code only, never the failed run's results or checkpoints.
    # Keep run_trial.ROOT unchanged for its canonical historical source paths.
    run_trial.OUT = HERE / "results"
    run_trial.main()