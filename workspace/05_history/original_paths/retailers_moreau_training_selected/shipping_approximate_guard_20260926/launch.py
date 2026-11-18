"""Named-tmux readiness supervisor with whole-child monotonic runtime."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from guard_common import ROOT,definitions,parent

definitions(globals(),parent.prior.previous.previous.PREVIOUS/'launch_gate.py',('save','main'),changes=(
    ("str(ROOT/'ap_gate.py')","str(ROOT/'preflight.py')"),
    ('whole AP gate child including imports and teardown; not a CCG run',
     'whole approximate-guard readiness child; no training or full CCG'),))

if __name__=='__main__':
    main()