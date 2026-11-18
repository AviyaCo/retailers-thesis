"""Repeat the frozen readiness grid with only native replay rejection removed."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from observed_common import (ROOT,PREVIOUS,PRODUCTION,PROTOCOL,core,np,torch,
    shipping_model,shipping_resources,metadata,read,check,digest,sources,verify,definitions)
import observed_adversary as adversary
from observed_pool import hard_objective,validate_ap,state_hash,pool_hash

definitions(globals(),PREVIOUS/'preflight.py',('error_info','main'),changes=((
    "for cwd,module in ((REPLAY,'test_epsilon'),(ROOT,'test_pool')):",
    "for cwd,module in ((ROOT,'test_observed'),):"),(
    "case.update(replay_exceptions=sum(r.get('consistency_exception',False) for r in case['samples']),",
    "case.update(replay_rejection_enabled=False,\n"
    "                            native_support_disagreements=sum(not r['activation_replay']['support_matches'] for r in case['samples'] if 'support_matches' in r.get('activation_replay',{})),\n"
    "                            native_epsilon_disagreements=sum(bool(r.get('activation_replay',{}).get('native_epsilon_violations_zero_based')) for r in case['samples']),\n"
    "                            native_cost_disagreements=sum(not r['replay']['cost_matches_previous_tolerance'] for r in case['samples'] if 'replay' in r),")))

if __name__=='__main__':
    raise SystemExit(main())