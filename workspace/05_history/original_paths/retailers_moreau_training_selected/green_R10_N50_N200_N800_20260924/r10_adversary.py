"""Private AP: retailer loops become 10; three cost branches stay THREE."""
from source_adapter import OLD, load_into

load_into(globals(), OLD / 'adversary.py', [
    ('import core', 'import r10_core as core'),
    ('first[:, 3:6]', 'first[:, 10:20]'),
    ('first[:, 6:]', 'first[:, 20:]'),
    ('first[:, :3]', 'first[:, :10]'),
    ('for r in range(3)', 'for r in range(10)'),
    ('x1h.shape != (3,) or x2h.shape != (3,)', 'x1h.shape != (10,) or x2h.shape != (10,)'),
    ('each have shape (3,)', 'each have shape (10,)'),
    ('solver.addVars(3,', 'solver.addVars(10,'),
])
# The distinct generator `for _ in range(3)` and NumericFocus=3 are untouched.