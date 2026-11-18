"""Use the frozen named-tmux supervisor with the readiness entrypoint only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from campaign_common import ROOT,definitions,previous

definitions(globals(),previous.PREVIOUS/'launch_gate.py',('save','main'),changes=(
    ("str(ROOT/'ap_gate.py')","str(ROOT/'preflight.py')"),
    ('whole AP gate child including imports and teardown; not a CCG run',
     'whole full-center readiness child including imports/teardown; not a CCG run'),))

if __name__=='__main__':
    main()