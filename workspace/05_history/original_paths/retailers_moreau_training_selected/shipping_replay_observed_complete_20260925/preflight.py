"""Unchanged readiness experiment; complete advisory cost counts on failure."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from common import (ROOT,PREVIOUS,PRODUCTION,PROTOCOL,core,np,torch,shipping_model,
    shipping_resources,metadata,read,check,digest,sources,verify,definitions,prior_pool)
import adversary

hard_objective,validate_ap=prior_pool.hard_objective,prior_pool.validate_ap
state_hash,pool_hash=prior_pool.state_hash,prior_pool.pool_hash
definitions(globals(),PREVIOUS/'preflight.py',('error_info','main'),changes=((
    "for cwd,module in ((REPLAY,'test_epsilon'),(ROOT,'test_pool')):",
    "for cwd,module in ((ROOT,'test_complete'),):"),(
    "case.update(replay_exceptions=sum(r.get('consistency_exception',False) for r in case['samples']),",
    "case.update(replay_rejection_enabled=False,\n"
    "                            native_support_disagreements=sum(not r['activation_replay']['support_matches'] for r in case['samples'] if 'support_matches' in r.get('activation_replay',{})),\n"
    "                            native_epsilon_disagreements=sum(bool(r.get('activation_replay',{}).get('native_epsilon_violations_zero_based')) for r in case['samples']),\n"
    "                            native_cost_observations=sum('native_cost_observation' in r for r in case['samples']),\n"
    "                            native_cost_disagreements=sum(not r['native_cost_observation']['cost_matches_previous_tolerance'] for r in case['samples'] if 'native_cost_observation' in r),")))

if __name__=='__main__':
    raise SystemExit(main())