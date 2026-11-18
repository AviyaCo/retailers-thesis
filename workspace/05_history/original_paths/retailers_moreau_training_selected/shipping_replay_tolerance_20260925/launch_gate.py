"""Retain the original named-tmux launcher and whole-child clock."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from epsilon_common import ROOT,PREVIOUS,definitions

definitions(globals(),PREVIOUS/'launch_gate.py',('save','main'))

if __name__=='__main__':
    main()