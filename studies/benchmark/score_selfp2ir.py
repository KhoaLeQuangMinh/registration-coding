"""
Score Self-P2IR predictions (from selfp2ir_predict.py) as pre-registered in PREREGISTRATION.md.

  python benchmark/score_selfp2ir.py --check   reproduction check only: the authors' registration
                                               recall on condition A, no Dice or TRE
  python benchmark/score_selfp2ir.py           H1, H2, H3 and E1 on every predictions_*.json found

Uses the pilot's code for Dice (tool pixels ignored), distance, the true-pose check and probe
tumours, so the numbers are computed exactly as in the pilot.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

import pilot_dice_vs_tumour as pilot
from pilot_dice_vs_tumour import (LIVER, SHAFT, TIP, apply, dice, load_patient, load_view, project,
                                  render_mask, sample_tumours, spearman, surface_dists, true_w2c)

N_PROBES = 200
MARGIN_MM = 10.0
DICE_ABS = 0.90
TOP_FRACTION = 0.25
N_BOOT = 2000
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def to_cv_mm(R, t) -> np.ndarray:
    """Their pose (CT metres -> Blender camera metres) as our 4x4 (CT mm -> OpenCV camera mm)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R)
    T[:3, 3] = np.asarray(t) * 1000.0
    return BLENDER_TO_CV @ T


def load_predictions(pred_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "predictions_*.json"))):
        d = json.load(open(path))
        if d["meta"].get("complete") is False:
            raise SystemExit(f"{path} is a partial run; finish it (rerun the same command) before scoring")
        out[d["meta"]["condition"]] = d
    assert "A" in out, f"no predictions_A.json in {pred_dir}"
    limits = {c: tuple(d["meta"]["neighbourhood_limits"]) for c, d in out.items()}
    assert len(set(limits.values())) == 1, f"conditions used different neighbourhood limits: {limits}"
    print("neighbourhood limits (all conditions):", next(iter(limits.values())))
    return out


# ==============================================================================
# Reproduction check (their metric only)
# ==============================================================================
def reproduction_check(preds):
    A = preds["A"]
    by_patient = {}
    for r in A["predictions"]:
        by_patient.setdefault(r["patient"], []).append(r)
    dists = []
    for pid, recs in by_patient.items():
        P = load_patient(f"{pid:02d}")
        verts = np.asarray(P["mesh"].vertices)
        for r in recs:
            T_pred, T_true = to_cv_mm(r["R"], r["t"]), to_cv_mm(r["R_true"], r["t_true"])
            dists.append(np.linalg.norm(apply(T_pred, verts) - apply(T_true, verts), axis=1).mean())
    dists = np.array(dists)
    print(f"Reproduction check, condition A: {len(dists)} views, {len(A['meta']['rejected'])} rejected by their loader")
    for rho in (2, 5, 10, 20, 40):
        print(f"  registration recall at rho = {rho:>2} mm: {100 * (dists < rho).mean():5.1f} %")
    print(f"  mean CT point error: median {np.median(dists):.2f} mm, 90th pct {np.percentile(dists, 90):.2f} mm")
    print("  paper (Fig. 7a): > 92 % at rho = 5 mm")


# ==============================================================================
# Per-view scoring
# ==============================================================================
def score_views(preds):
    """One row per (condition, view) that passes the true-pose check in every condition."""
    patients = sorted({r["patient"] for d in preds.values() for r in d["predictions"]})
    rows, excluded = [], []
    for pid in patients:
        P = load_patient(f"{pid:02d}")
        probes, _ = sample_tumours(P, np.random.default_rng([0, pid]), N_PROBES)
        views = sorted({r["view"] for r in preds["A"]["predictions"] if r["patient"] == pid})
        recs = {c: {r["view"]: r for r in d["predictions"] if r["patient"] == pid} for c, d in preds.items()}

        for view in views:
            label, vis_pts = load_view(P, view)
            T_true = true_w2c(P["poses"][f"{view:05d}"][0])
            ignore = (label == TIP) | (label == SHAFT)
            d0 = dice(render_mask(T_true, P), label == LIVER, ignore)
            far0 = (surface_dists(T_true, P, vis_pts) > 3.0).mean()
            if d0 < 0.98 or far0 > 0.02 or any(view not in recs[c] for c in preds):
                excluded.append((pid, view, d0, far0))
                continue

            # Their ground truth must be the same pose as ours
            T_their = to_cv_mm(recs["A"][view]["R_true"], recs["A"][view]["t_true"])
            assert np.abs(T_their - T_true).max() < 1e-2, f"pose convention mismatch at {pid}/{view}"

            gt_mask = render_mask(T_true, P)
            uv = project(apply(T_true, probes), P["K"])
            ok = np.isfinite(uv).all(axis=1)
            u, v = np.full(len(uv), -1), np.full(len(uv), -1)
            u[ok], v[ok] = np.round(uv[ok, 0]).astype(int), np.round(uv[ok, 1]).astype(int)
            inb = ok & (u >= 0) & (u < gt_mask.shape[1]) & (v >= 0) & (v < gt_mask.shape[0])
            in_view = np.zeros(len(probes), bool)
            in_view[inb] = gt_mask[v[inb], u[inb]]

            c_true = apply(T_true, probes)
            for cond in preds:
                r = recs[cond][view]
                T_pred = to_cv_mm(r["R"], r["t"])
                c_pred = apply(T_pred, probes)
                err = np.linalg.norm(c_pred - c_true, axis=1)
                rows.append(dict(
                    condition=cond, patient=pid, view=view, scale=r["scale"],
                    dice=dice(render_mask(T_pred, P), label == LIVER, ignore),
                    distance_mm=float(surface_dists(T_pred, P, vis_pts * r["scale"]).mean()),
                    view_tre_mm=float(err.mean()),
                    tre=err, depth=np.abs(c_pred[:, 2] - c_true[:, 2]),
                    px=np.linalg.norm(project(c_pred, P["K"]) - project(c_true, P["K"]), axis=1),
                    in_view=in_view))
            print(f"[{pid:02d}] view {view:05d} scored")
    return rows, excluded


# ==============================================================================
# Hypotheses
# ==============================================================================
def boot_ci(rows, stat):
    """95 % interval from resampling patients (the independent units)."""
    pids = sorted({r["patient"] for r in rows})
    by = {p: [r for r in rows if r["patient"] == p] for p in pids}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(N_BOOT):
        sample = [r for p in rng.choice(pids, len(pids)) for r in by[p]]
        s = stat(sample)
        if np.isfinite(s):
            vals.append(s)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def good_sets(rows):
    d = np.array([r["dice"] for r in rows])
    return {"Dice >= 0.90": d >= DICE_ABS,
            "top 25 % Dice": d >= np.quantile(d, 1 - TOP_FRACTION)}


def miss_rate(rows, margin=MARGIN_MM):
    """% of (view, probe) pairs with TRE above the margin."""
    if not rows:
        return np.nan
    return 100 * (np.concatenate([r["tre"] for r in rows]) > margin).mean()


def h1(rows, lines):
    rates = {}
    for name, m in good_sets(rows).items():
        good = [r for r, g in zip(rows, m) if g]
        rate = miss_rate(good)
        lo, hi = boot_ci(rows, lambda s, name=name: miss_rate(
            [r for r, g in zip(s, good_sets(s)[name]) if g]))
        per_view = 100 * np.mean([r["view_tre_mm"] > MARGIN_MM for r in good]) if good else np.nan
        curve = " ".join(f"{mg}:{miss_rate(good, mg):.0f}" for mg in range(0, 11, 2)) if good else "-"
        lines.append(f"  H1 [{name}] {len(good)} views: {rate:.1f} % of probes > 10 mm (95 % CI {lo:.1f}-{hi:.1f}); "
                     f"> 5 mm {miss_rate(good, 5):.1f} %; per-view > 10 mm {per_view:.1f} %")
        lines.append(f"     % of probes over margin (mm:%): {curve}")
        rates[name] = rate
    r = list(rates.values())
    if all(np.isfinite(r)) and all(x >= 10 for x in r):
        verdict = "a good Dice is NOT enough"
    elif all(np.isfinite(r)) and all(x < 2 for x in r):
        verdict = "a good Dice IS enough for this model"
    else:
        verdict = "INCONCLUSIVE"
    lines.append(f"  H1 verdict: {verdict}")


def h2(rows, lines):
    stat = lambda s: spearman(np.array([r["dice"] for r in s]), np.array([r["view_tre_mm"] for r in s]))
    rho = stat(rows)
    lo, hi = boot_ci(rows, stat)
    a = abs(rho)
    verdict = "good proxy" if a >= 0.8 else ("partial" if a >= 0.4 else "weak") if np.isfinite(a) else "n/a"
    lines.append(f"  H2 Spearman(Dice, view TRE) = {rho:+.2f} (95 % CI {lo:+.2f} to {hi:+.2f}) -> {verdict}")


def e1(rows, lines):
    tre = np.array([r["view_tre_mm"] for r in rows])
    lines.append(f"  E1 Spearman with view TRE: Dice {spearman(np.array([r['dice'] for r in rows]), tre):+.2f}, "
                 f"distance {spearman(np.array([r['distance_mm'] for r in rows]), tre):+.2f} (exploratory)")


def extras(rows, lines):
    """Also promised by the pre-registration: on-screen part, probes in view only, per patient. No verdicts."""
    cat = lambda key, sel: np.concatenate([r[key][r["in_view"]] if sel else r[key] for r in rows])
    good = [r for r in rows if r["dice"] >= DICE_ABS]
    in_good = np.concatenate([r["tre"][r["in_view"]] for r in good]) if good else np.array([])
    lines.append(f"  also: on-screen part median {np.nanmedian(cat('px', False)):.1f} px; "
                 f"probes in view {100 * np.mean(cat('in_view', False)):.0f} %, their median TRE "
                 f"{np.median(cat('tre', True)):.1f} mm; Dice >= 0.90 views, in-view probes > 10 mm: "
                 f"{100 * (in_good > MARGIN_MM).mean() if in_good.size else np.nan:.1f} %")
    for pid in sorted({r["patient"] for r in rows}):
        rp = [r for r in rows if r["patient"] == pid]
        gp = [r for r in rp if r["dice"] >= DICE_ABS]
        lines.append(f"     patient {pid:02d}: {len(rp)} views, median Dice {np.median([r['dice'] for r in rp]):.3f}, "
                     f"median view TRE {np.median([r['view_tre_mm'] for r in rp]):.1f} mm, "
                     f"Dice >= 0.90 probes > 10 mm {miss_rate(gp):.1f} % ({len(gp)} views)")


def h3(by_cond, lines):
    A = {(r["patient"], r["view"]): r for r in by_cond["A"]}
    for cond, rows in by_cond.items():
        if cond == "A":
            continue
        pairs = [(A[(r["patient"], r["view"])], r) for r in rows if (r["patient"], r["view"]) in A]
        dd = np.array([b["dice"] - a["dice"] for a, b in pairs])
        dt = np.array([b["view_tre_mm"] - a["view_tre_mm"] for a, b in pairs])
        confirmed = abs(np.median(dd)) < 0.02 and np.median(dt) > 5.0
        lines.append(f"  H3 [{cond}] {len(pairs)} paired views: median change in Dice {np.median(dd):+.3f} "
                     f"(median |change| {np.median(np.abs(dd)):.3f}), median change in view TRE {np.median(dt):+.1f} mm "
                     f"-> {'CONFIRMED (Dice is blind)' if confirmed else 'not confirmed'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="benchmark/selfp2ir_out")
    ap.add_argument("--data-root", default=pilot.DATA_DIR)
    ap.add_argument("--check", action="store_true", help="reproduction check only")
    args = ap.parse_args()
    pilot.DATA_DIR = args.data_root  # read by load_patient
    preds = load_predictions(args.pred_dir)

    if args.check:
        reproduction_check(preds)
        return

    rows, excluded = score_views(preds)
    by_cond = {c: [r for r in rows if r["condition"] == c] for c in preds}

    lines = [f"Self-P2IR, pre-registered analysis ({len(by_cond['A'])} views per condition; "
             f"{len(excluded)} excluded by the true-pose check or missing in a condition; "
             f"loader-rejected: {len(preds['A']['meta']['rejected'])})"]
    for cond, rows_c in by_cond.items():
        tre = np.concatenate([r["tre"] for r in rows_c])
        lines.append(f"\n== condition {cond}: median Dice {np.median([r['dice'] for r in rows_c]):.3f}, "
                     f"median probe TRE {np.median(tre):.1f} mm (depth part "
                     f"{np.median(np.concatenate([r['depth'] for r in rows_c])):.1f} mm) ==")
        h1(rows_c, lines)
        h2(rows_c, lines)
        e1(rows_c, lines)
        extras(rows_c, lines)
    lines.append("\n== H3: does Dice notice a wrong scale? ==")
    h3(by_cond, lines)

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.pred_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.pred_dir, "per_view.csv"), "w", newline="") as f:
        keys = ["condition", "patient", "view", "scale", "dice", "distance_mm", "view_tre_mm"]
        w = csv.DictWriter(f, fieldnames=keys + ["probes_over_10mm_pct", "median_depth_mm", "median_px",
                                                 "probes_in_view", "in_view_tre_mm"])
        w.writeheader()
        for r in rows:
            iv = r["in_view"]
            w.writerow({**{k: r[k] for k in keys}, "probes_over_10mm_pct": 100 * (r["tre"] > MARGIN_MM).mean(),
                        "median_depth_mm": np.median(r["depth"]), "median_px": np.nanmedian(r["px"]),
                        "probes_in_view": int(iv.sum()), "in_view_tre_mm": r["tre"][iv].mean() if iv.any() else ""})
    with open(os.path.join(args.pred_dir, "excluded_views.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "view", "true_pose_dice", "far_point_fraction"])
        w.writerows(excluded)


if __name__ == "__main__":
    main()
