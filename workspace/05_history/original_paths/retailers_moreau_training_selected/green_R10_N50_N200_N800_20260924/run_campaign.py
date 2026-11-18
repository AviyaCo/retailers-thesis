"""Fresh isolated R10 campaign. Import-safe; production requires explicit CLI."""
from source_adapter import GREEN, SOURCE_NAMES, load_into

load_into(globals(), GREEN / 'run_campaign.py', [
    ('centers[:,:3]', 'centers[:,:10]'),
    ('centers[:,3:]', 'centers[:,10:]'),
    ("centers.shape==(record['Ntrain'],6)", "centers.shape==(record['Ntrain'],20)"),
    ('workers=min(cpu_slots,mem_slots,180)', 'workers=min(cpu_slots,mem_slots,270)'),
    ('min(workers,60)', 'min(workers,90)'),
    ('min(workers,20)', 'min(workers,30)'),
    ("required_sources = ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py',\n                            'run_campaign.py', 'test_campaign.py', 'preflight.py', 'launch.sh', 'README.md')",
     'required_sources = SOURCE_NAMES'),
    ("check(preflight['status'] == 'completed', 'Require completed real preflight')",
     "check(preflight['status'] == 'completed', 'Require completed real preflight')\n        check(preflight.get('frozen_sources_sha256') == FROZEN, 'Preflight historical binding mismatch')"),
])
from r10_resources import resources

if __name__ == '__main__':
    sys.exit(main())