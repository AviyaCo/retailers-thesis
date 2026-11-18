"""Same reserve policy, using all mounted cgroup ancestors, not just roots."""
from source_adapter import GREEN, load_into, THREAD_ENV
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import resource
import sys

load_into(globals(), GREEN / 'preflight.py', names=('utc', 'peak_rss_bytes', 'environment'))

def _cpus(text):
    result = set()
    for part in text.split(','):
        a, _, b = part.partition('-')
        result.update(range(int(a), int(b or a) + 1))
    return result

def resources():
    snapshot = environment()
    files = snapshot['cgroup_files']
    quotas, memory = [], []
    affinity = set(snapshot['affinity'])
    for path, text in files.items():
        path = Path(path)
        if path.name == 'cpu.max':
            quota, period = text.split()
            if quota != 'max':
                quotas.append(int(quota) / int(period))
        elif path.name == 'cpu.cfs_quota_us' and int(text) > 0:
            quotas.append(int(text) / int(files[str(path.with_name('cpu.cfs_period_us'))]))
        elif path.name in ('memory.max', 'memory.limit_in_bytes') and text != 'max':
            usage_name = 'memory.current' if path.name == 'memory.max' else 'memory.usage_in_bytes'
            memory.append(max(0, int(text) - int(files[str(path.with_name(usage_name))])))
        elif path.name in ('cpuset.cpus.effective', 'cpuset.cpus') and text:
            affinity &= _cpus(text)
    return dict(host=snapshot['hostname'], platform=snapshot['platform'], python=sys.version,
        executable=sys.executable, logical_cpus=os.cpu_count(), affinity=sorted(affinity),
        load_average=snapshot['loadavg'], cpu_quota_cores=min(quotas) if quotas else None,
        memory_available_bytes=min([snapshot['memory']['MemAvailable_bytes'], *memory]),
        host_memory=snapshot['memory'], cgroup_memory_headrooms_bytes=memory,
        cgroup=snapshot['cgroup_membership'], cgroup_files=files,
        captured_at_utc=snapshot['at_utc'])