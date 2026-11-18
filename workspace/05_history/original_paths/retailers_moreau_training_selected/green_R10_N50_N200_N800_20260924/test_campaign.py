"""Full original campaign regression suite; held-out mock shapes adapted to R10."""
from source_adapter import GREEN, load_into
load_into(globals(), GREEN / 'test_campaign.py', [
    ("[[1.,1.,1.]]*40", "[[1.]*10]*40"),
    ("[[2.,2.,2.]]*40", "[[2.]*10]*40"),
])
if __name__ == '__main__':
    unittest.main()