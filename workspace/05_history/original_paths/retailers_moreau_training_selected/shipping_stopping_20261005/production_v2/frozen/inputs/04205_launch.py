"""Run the unchanged tmux supervisor with this revision's working directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import ast

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'shipping_comparison_20260928/launch.py'
nodes = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main']
if len(nodes) != 1:
    raise ValueError('Ambiguous parent launcher')
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), globals())

if __name__ == '__main__':
    main()