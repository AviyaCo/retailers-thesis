"""Repeat all 1680 training-center cases under the authorized approximate guards."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from guard_common import (ROOT,PRODUCTION,PROTOCOL,core,np,torch,shipping_model,
    shipping_resources,metadata,read,check,digest,sources,verify,definitions,parent)
import guard_adversary as adversary
from guard_pool import hard_objective,validate_ap,state_hash,pool_hash

definitions(globals(),parent.PREVIOUS/'preflight.py',('error_info','main'),changes=((
    "for cwd,module in ((REPLAY,'test_epsilon'),(ROOT,'test_pool')):",
    "for cwd,module in ((ROOT,'test_guards'),):"),(
    "case.update(replay_exceptions=sum(r.get('consistency_exception',False) for r in case['samples']),",
    "case.update(replay_rejection_enabled=False,\n"
    "                            material_negative_bound_pool_gaps=sum(r.get('material_negative_bound_pool_gap',False) for r in case['samples']),\n"
    "                            native_support_disagreements=sum(not r['activation_replay']['support_matches'] for r in case['samples'] if 'support_matches' in r.get('activation_replay',{})),\n"
    "                            native_epsilon_disagreements=sum(bool(r.get('activation_replay',{}).get('native_epsilon_violations_zero_based')) for r in case['samples']),\n"
    "                            native_cost_observations=sum('native_cost_observation' in r for r in case['samples']),\n"
    "                            native_cost_disagreements=sum(not r['native_cost_observation']['cost_matches_previous_tolerance'] for r in case['samples'] if 'native_cost_observation' in r),")))

if __name__=='__main__':
    raise SystemExit(main())