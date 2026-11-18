"""Read-only adapters for tested mounted-cgroup/affinity resource accounting."""
from source_adapter import GREEN, RESOURCE, load_into, THREAD_ENV, check
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import resource
import sys

load_into(globals(), GREEN/'preflight.py', ('utc','peak_rss_bytes','environment'))
load_into(globals(), RESOURCE, ('_cpus','resources'))


def choose_workers(snapshot, peak):
    effective = len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective = min(effective, math.floor(snapshot['cpu_quota_cores']))
    background = math.ceil(max(snapshot['load_average']))
    cpu_slots = effective-background-4
    budget = max(4*1024**3,2*peak)
    mem_slots = (snapshot['memory_available_bytes']-16*1024**3)//budget
    slots = int(min(cpu_slots,mem_slots))
    check(slots >= 1, 'Insufficient headroom: abort, no degraded launch')
    return dict(initialization=min(30,slots), nominal=min(90,slots),
        reserve_cpu_slots=4, reserved_background_cpu=background,
        memory_budget_bytes_per_worker=budget, host_memory_reserve_bytes=16*1024**3,
        observed_preflight_peak_rss_bytes=peak)