"""Load only definitions from the canonical script, never its launch pipeline."""
import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.stats import truncnorm
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parent.parent / "retailers_problem/R3/N50/saainit_net16_v2/run_all_net16_v2.py"
os.environ.setdefault("GRB_LICENSE_FILE", "/data/home/aviyac/gurobi.lic")


def load_baseline():
    # The original script parses CLI arguments, creates directories, and runs
    # experiments at import time. Compile its exact selected definitions only.
    names = {"Cfg", "softplus_inverse", "solve_saa", "generate_data", "Model",
             "make_model_saa_init", "train"}
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError("canonical source definitions changed")
    scope = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    cfg = scope["Cfg"]()
    cfg.LR_NET = .05
    scope["cfg"] = cfg
    return SimpleNamespace(**{k: scope[k] for k in names}, cfg=cfg)