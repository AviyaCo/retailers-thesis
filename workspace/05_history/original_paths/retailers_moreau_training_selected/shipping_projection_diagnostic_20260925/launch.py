"""Frozen named-tmux supervisor with the diagnostic entrypoint."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from common import ROOT,parent

parent.definitions(globals(),parent.previous.PREVIOUS/'launch_gate.py',('save','main'),changes=(
    ("str(ROOT/'ap_gate.py')","str(ROOT/'diagnose.py')"),
    ('whole AP gate child including imports and teardown; not a CCG run',
     'whole projection diagnostic child including imports/teardown; no training or CCG'),))

if __name__=='__main__':
    main()