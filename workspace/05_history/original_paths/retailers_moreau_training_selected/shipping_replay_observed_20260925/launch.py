"""Frozen named-tmux supervisor; no full CCG launch."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from observed_common import ROOT,definitions,previous

definitions(globals(),previous.previous.PREVIOUS/'launch_gate.py',('save','main'),changes=(
    ("str(ROOT/'ap_gate.py')","str(ROOT/'preflight.py')"),
    ('whole AP gate child including imports and teardown; not a CCG run',
     'whole readiness child with advisory native replay; no training or CCG'),))

if __name__=='__main__':
    main()