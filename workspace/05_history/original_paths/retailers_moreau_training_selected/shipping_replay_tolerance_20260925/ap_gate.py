"""Same six training-only checks; explicitly distinguish tolerance exceptions."""
import argparse
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from epsilon_common import (ROOT,PREVIOUS,PRODUCTION,PROTOCOL,core,torch,np,read,
    check,digest,sources,verify,metadata,shipping_model,shipping_resources,definitions)
from epsilon_adversary import solve_ap

definitions(globals(),PREVIOUS/'ap_gate.py',('main',),changes=(
    ("['accepted']=True","['accepted']=True\n                    row['acceptance_status']=row['replay_consistency']"),
    ("row.update(accepted=False,error=dict", "row.update(accepted=False,acceptance_status='failed',error=dict"),
    ("        record['status']='completed'",
     "        record['ordinary_pass_count']=sum(c['acceptance_status']=='passed' for c in record['cases'])\n"
     "        record['tolerated_exception_count']=sum(c['consistency_exception'] for c in record['cases'])\n"
     "        record['all_replays_consistent']=record['tolerated_exception_count']==0\n"
     "        record['status']='completed'"),
))

if __name__=='__main__':
    raise SystemExit(main())