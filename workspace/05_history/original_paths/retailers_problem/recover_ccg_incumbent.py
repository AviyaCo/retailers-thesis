"""
recover_ccg_incumbent.py  --  Repoint corrupted CCG summaries to their best-AP incumbent.

Background
----------
The CCG workers (run_rho_seed*.py) had two coupled bugs:
  1. Convergence was declared whenever  dpct = (AP - MP)/|MP| < CCG_CONV,
     which also fires for a large NEGATIVE gap -> a diverged master is
     mistaken for convergence.
  2. The summary saved the LAST iterate (final_mp_iter), not the best one.

Because the master objective rises by construction in CCG, the right quality
signal is the ADVERSARIAL (AP) value: the policy with the lowest worst-case AP
is the genuine incumbent. This script does NOT retrain; every per-iteration
checkpoint already exists on disk. It simply:

  - finds  best_it = argmin_iter ap_mean   (lowest worst-case)
  - the incumbent policy is the one that AP evaluated at best_it, i.e.
        incumbent_mp_iter = best_it - 1     (0 == nominal policy)
  - if that differs from final_mp_iter and the saved policy is materially worse
    (AP regret > THRESH%) or convergence was declared on a negative gap,
    it rewrites the summary to point at the incumbent and records the reason.

After running with --apply, re-run the corresponding eval_sweep_*.py with
--force-rhos <changed rhos> so IS/OOS are recomputed from the incumbent.
"""

import argparse, glob, json, os, shutil

DIRS = {
    "R3/N50":       "retailers_problem/R3/N50/rho_sweep_net16_v2",
    "R3/N200":      "retailers_problem/R3/N200/rho_sweep_net16_v2",
    "R3/N800":      "retailers_problem/R3/N800/rho_sweep_net16_v2",
    "R10/N50":      "retailers_problem/R10/N50/rho_sweep_net16_v2",
    "R10/N200":     "retailers_problem/R10/N200/rho_sweep_net16_v2",
    "R10/N800":     "retailers_problem/R10/N800/rho_sweep_net16_v2",
    "shipping/N50": "retailers_with_shipping/N50/rho_sweep",
    "shipping/N200": "retailers_with_shipping/N200/rho_sweep",
    "shipping/N800": "retailers_with_shipping/N800/rho_sweep",
}
ROOT = "/data/home/aviyac"
THRESH = 1.0  # AP-regret %% above which a non-incumbent saved policy is "affected"


def analyse(summary):
    """Return (incumbent_mp_iter, best_ap, ap_regret_pct, neg_gap) or None."""
    tr = summary.get("ccg_trajectory")
    if not isinstance(tr, list) or not tr:
        return None
    aps = {e["iter"]: e["ap_mean"] for e in tr
           if "ap_mean" in e and "iter" in e}
    if not aps:
        return None
    fmp = summary.get("final_mp_iter", 0)
    best_it = min(aps, key=aps.get)
    best_ap = aps[best_it]
    inc_mp = best_it - 1
    ap_final = aps.get(fmp + 1)  # AP value of the saved policy
    regret = (ap_final - best_ap) / best_ap * 100 if ap_final else None
    neg = any(e.get("converged") and e.get("delta_pct", 0) < -0.5 for e in tr)
    return inc_mp, best_ap, regret, neg


def mp_loss_at(summary, it):
    if it <= 0:
        return summary.get("nominal_loss")
    for e in summary["ccg_trajectory"]:
        if e.get("iter") == it and "mp_loss" in e:
            return e["mp_loss"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default: dry run).")
    ap.add_argument("--groups", nargs="+", choices=DIRS, default=list(DIRS),
                    help="Limit recovery to the listed experiment groups.")
    ap.add_argument("--all-mismatches", action="store_true",
                    help="Repoint every non-incumbent summary, regardless of AP regret.")
    args = ap.parse_args()

    changed_rhos = {}  # dir name -> set of rho values changed
    total = 0
    for name, rel in DIRS.items():
        if name not in args.groups:
            continue
        root = os.path.join(ROOT, rel)
        rows = []
        for f in sorted(glob.glob(f"{root}/**/seed_*_summary.json", recursive=True)):
            try:
                s = json.load(open(f))
            except Exception:
                continue
            res = analyse(s)
            if res is None:
                continue
            inc_mp, best_ap, regret, neg = res
            fmp = s.get("final_mp_iter", 0)
            affected = (inc_mp != fmp) and (
                args.all_mismatches
                or (regret is not None and regret > THRESH)
                or neg)
            if not affected:
                continue
            rows.append((s.get("rho"), s.get("seed"), fmp, inc_mp, regret, neg, f, s, best_ap))

        if rows:
            print(f"\n=== {name}: {len(rows)} affected ===")
        for rho, seed, fmp, inc_mp, regret, neg, f, s, best_ap in sorted(rows, key=lambda x: (x[0], x[1])):
            reg_s = f"{regret:.1f}%" if regret is not None else "n/a"
            print(f"   rho={rho} seed={seed}: mp_iter{fmp} -> mp_iter{inc_mp} "
                  f"(AP_regret={reg_s}, neg_gap={neg})")
            total += 1
            changed_rhos.setdefault(name, set()).add(rho)
            if args.apply:
                backup = f + ".pre_recovery"
                if not os.path.exists(backup):
                    shutil.copy2(f, backup)
                new_loss = mp_loss_at(s, inc_mp)
                s["recovery"] = {
                    "applied": True,
                    "orig_final_mp_iter": fmp,
                    "orig_final_mp_loss": s.get("final_mp_loss"),
                    "orig_converged": s.get("converged"),
                    "incumbent_mp_iter": inc_mp,
                    "incumbent_ap_value": best_ap,
                    "ap_regret_pct": regret,
                    "neg_gap_convergence": neg,
                    "note": (
                        "The reported incumbent is the generated policy with the "
                        "lowest empirical mean adversarial (AP) cost. AP iteration "
                        "t evaluates MP iteration t-1, with iteration 1 evaluating "
                        "the nominal policy. final_mp_iter, ccg_iters, and "
                        "final_mp_loss were repointed to mp_iter{}. No retraining "
                        "was done; the checkpoint already existed on disk."
                    ).format(inc_mp)
                }
                s["best_ap_value"] = best_ap
                s["best_ap_iter"] = inc_mp + 1
                s.setdefault("last_mp_iter", fmp)
                s.setdefault("last_mp_loss", s.get("final_mp_loss"))
                s["final_mp_iter"] = inc_mp
                s["ccg_iters"] = inc_mp
                if new_loss is not None:
                    s["final_mp_loss"] = new_loss
                if neg:
                    s["converged"] = False
                with open(f, "w") as fh:
                    json.dump(s, fh, indent=2)

    print(f"\nTOTAL affected: {total}")
    print("\nRe-eval commands (run after --apply):")
    for name, rhos in changed_rhos.items():
        rel = DIRS[name]
        ev = sorted(glob.glob(os.path.join(ROOT, rel, "eval_sweep_*.py")))
        ev = [e for e in ev if "10k" in e or "net16_v2" in e or "shipping" in e]
        evname = os.path.basename(ev[-1]) if ev else "eval_sweep_*.py"
        rho_args = " ".join(f"{r:g}" for r in sorted(rhos))
        print(f"  ({name})  cd {rel} && python {evname} --force-rhos {rho_args}")


if __name__ == "__main__":
    main()
