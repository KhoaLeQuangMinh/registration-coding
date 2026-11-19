"""
Does a ridge help mask-only registration beyond searching harder, and what ambiguity does the outline leave?

On clean synthetic training views (true pose known), outline_reg.py registers the CT from scratch:
  outline       liver mask only; 40 viewing directions x 8 rolls, 10 candidates refined fully
  ridge         the same + a perfect ridge: the CT ridge (ct_landmarks.py, frozen) seen at the true
                pose (in view, not hidden, not under a tool), projected into the image, weight 1
  outline_more  liver mask only, a 4x bigger search: 80 x 16 starts, 40 candidates refined
"Perfect" means a detector that marks exactly our 3D rule's ridge; it measures how much a right
ridge could add, not how good our rule or a detector is. outline_more is the control: on
patients 01, 06, 10 and 11 most failures were search misses, which more search might fix as well.

Scored by the mean error at the patient's 200 probe tumours (as capture_range.py); success < 10 mm
(the LLR-LUS inclusion margin), < 5 mm also reported.

Each failure (>= 10 mm) is classified with T*, the pose refined from the true pose:
  search       fit(F) > fit(T*) + 0.5 px: a better-fitting pose exists, the search missed it
  alternative  otherwise: the outline fits the wrong pose F as well as the right one; then the
               path from T* to F (rotation and camera centre interpolated, 9 steps, shift
               re-fitted at each) decides
                 separate   fit rises > 1 px above both ends somewhere on the path: a distinct wrong pose
                 connected  otherwise: a smooth valley of poses that fit equally
T* itself drifting >= 10 mm from the truth is a connected valley at the truth.
Views whose mask has < 20 outline pixels on the coarse grid are skipped (no outline to fit).

Run 1 (2026-11-10, patients 01 06 10 11, 20 views each, outline and ridge only) was exploratory:
the design was built and looked at on those patients. Run 2 (all P2I-LReg patients, 10 views each,
all three conditions; fixed on 2026-11-11 before it started) is confirmatory:
  primary   success(ridge) - success(outline_more) on the 17 patients other than 01 06 10 11,
            paired per view; 95 % CI from 2000 bootstrap resamples of patients (seed 0)
            CI above 0: the ridge adds beyond search | below 0: the ridge hurts | otherwise: inconclusive
  secondary the same for ridge - outline and outline_more - outline; failure kinds per condition;
            all of it again for patients 01 06 10 11, reported separately

Usage: python registration/test_ridge_value.py [--data-root dataset/liver-registration] [--patients 01 06 ...]
           [--views-per-patient 10] [--workers 4] [--out registration/out/ridge_value.csv] [--summary-only]
The CSV is written row by row; a rerun skips the (patient, view, condition) rows already in it.
"""
import argparse
import csv
import json
import os
import multiprocessing

import numpy as np

import pilot_dice_vs_tumour as pilot
from capture_range import HERE, N_PROBES, training_views
from ct_landmarks import ridge
from outline_reg import fit_error, make_view, project, register, register_from_scratch, visible
from pilot_dice_vs_tumour import LIVER, SHAFT, TIP, load_patient, load_view, rotation_deg, sample_tumours
from sim3_icp import rotation

CONDITIONS = {  # name -> (ridge used, viewing directions, rolls, candidates refined fully)
    "outline": (False, 40, 8, 10),
    "ridge": (True, 40, 8, 10),
    "outline_more": (False, 80, 16, 40),
}
DEV_PATIENTS = ["01", "06", "10", "11"]
SUCCESS_MM = 10.0
ALT_PX, BARRIER_PX, PATH_STEPS = 0.5, 1.0, 9
FIELDS = ["patient", "view", "condition", "ridge_points_seen", "error_mm", "rotation_error_deg", "fit_px",
          "truth_refined_error_mm", "truth_refined_fit_px", "failure", "barrier_px"]
_patients = {}


def path_barrier(scene, view, A, B, ridge3d):
    """Highest fit on the way from pose A to pose B, above the higher end (px)."""
    (Ra, ta), (Rb, tb) = A, B
    Ca, Cb = -Ra.T @ ta, -Rb.T @ tb
    rel = Ra @ Rb.T
    angle = rotation_deg(rel)
    w = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    axis = w / (np.linalg.norm(w) + 1e-12)
    worst = 0.0
    for k in np.arange(1, PATH_STEPS + 1) / (PATH_STEPS + 1):
        R = rotation(axis, -k * angle) @ Ra  # Ra -> Rb
        R, t = register(scene, view, R, -R @ ((1 - k) * Ca + k * Cb), ridge3d, iters=20, rotate=False)
        worst = max(worst, fit_error(scene, view, R, t, ridge3d))
    return worst - max(fit_error(scene, view, *A, ridge3d), fit_error(scene, view, *B, ridge3d))


def ridge_points(P, pid):
    """The patient's CT ridge vertices (mm); computed with the frozen rule and saved if not there yet."""
    path = os.path.join(HERE, "landmarks", f"{pid}.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"ridge": [p.tolist() for p in ridge(P["mesh"])]}, open(path, "w"))
    return np.asarray(P["mesh"].vertices)[np.concatenate(json.load(open(path))["ridge"])]


def patient(data_root, pid):
    if pid not in _patients:
        pilot.DATA_DIR = data_root
        P = load_patient(pid)
        P["ridge3d"] = ridge_points(P, pid)
        P["probes"] = sample_tumours(P, np.random.default_rng([0, int(pid)]), N_PROBES)[0]
        _patients.clear()  # one patient in memory per worker
        _patients[pid] = P
    return _patients[pid]


def run_view(task):
    """All missing conditions for one view: list of CSV rows."""
    data_root, pid, view_id, T, conditions = task
    P = patient(data_root, pid)
    label, _ = load_view(P, view_id)
    R0, t0 = T[:3, :3], T[:3, 3]
    view = make_view(P["K"], label == LIVER, (label == TIP) | (label == SHAFT))
    if len(view["liver_edge"]) < 20:
        return []
    seen = P["ridge3d"][visible(P["scene"], view, R0, t0, P["ridge3d"])]
    view["ridge2d"] = project(P["K"], seen @ R0.T + t0)
    probes = P["probes"]
    err = lambda R, t: np.linalg.norm(probes @ R.T + t - (probes @ R0.T + t0), axis=1).mean()  # noqa: E731

    rows = []
    for cond in conditions:
        use_ridge, n_dirs, n_rolls, keep = CONDITIONS[cond]
        rid = P["ridge3d"] if use_ridge else None
        T_star = register(P["scene"], view, R0, t0, rid)
        F, fit_F = register_from_scratch(P["scene"], view, P["centroid"], rid, n_dirs=n_dirs, n_rolls=n_rolls, keep=keep)
        fit_star = fit_error(P["scene"], view, *T_star, rid)
        row = dict(patient=pid, view=view_id, condition=cond, ridge_points_seen=len(seen), error_mm=err(*F),
                   rotation_error_deg=rotation_deg(F[0] @ R0.T), fit_px=fit_F, truth_refined_error_mm=err(*T_star),
                   truth_refined_fit_px=fit_star, failure="", barrier_px="")
        if row["error_mm"] >= SUCCESS_MM:
            if fit_F > fit_star + ALT_PX:
                row["failure"] = "search"
            else:
                row["barrier_px"] = path_barrier(P["scene"], view, T_star, F, rid)
                row["failure"] = "separate" if row["barrier_px"] > BARRIER_PX else "connected"
        rows.append(row)
    return rows


def bootstrap_ci(diff_by_patient, n=2000, seed=0):
    """95 % CI of the mean per-view difference, resampling patients."""
    groups = list(diff_by_patient.values())
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault((r["patient"], int(r["view"])), {})[r["condition"]] = r
    for name, group in (("17 new patients (confirmatory)", lambda p: p not in DEV_PATIENTS),
                        ("patients 01 06 10 11 (development)", lambda p: p in DEV_PATIENTS)):
        views = {k: v for k, v in by.items() if group(k[0]) and len(v) == len(CONDITIONS)}
        if not views:
            continue
        print(f"\n{name}: {len(views)} views, {len({k[0] for k in views})} patients")
        for cond in CONDITIONS:
            e = np.array([float(v[cond]["error_mm"]) for v in views.values()])
            kinds = {k: sum(v[cond]["failure"] == k for v in views.values()) for k in ("search", "separate", "connected")}
            drift = sum(float(v[cond]["truth_refined_error_mm"]) >= SUCCESS_MM for v in views.values())
            print(f"  {cond:12s} < {SUCCESS_MM:.0f} mm {100 * (e < SUCCESS_MM).mean():5.1f} %  < 5 mm {100 * (e < 5).mean():5.1f} %  "
                  f"median {np.median(e):5.1f} mm  failures {kinds}  truth drifts: {drift}")
        for a, b in (("ridge", "outline_more"), ("ridge", "outline"), ("outline_more", "outline")):
            diff = {}
            for (pid, _), v in views.items():
                ok = lambda c: float(v[c]["error_mm"]) < SUCCESS_MM  # noqa: E731
                diff.setdefault(pid, []).append(float(ok(a)) - float(ok(b)))
            diff = {p: np.array(d) for p, d in diff.items()}
            mean = np.concatenate(list(diff.values())).mean()
            lo, hi = bootstrap_ci(diff)
            fixed = sum((d > 0).sum() for d in diff.values())
            broke = sum((d < 0).sum() for d in diff.values())
            verdict = "adds" if lo > 0 else "hurts" if hi < 0 else "inconclusive"
            print(f"  {a} - {b}: {100 * mean:+.1f} points (95 % CI {100 * lo:+.1f} to {100 * hi:+.1f}), "
                  f"{fixed} views fixed, {broke} broken -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=pilot.DATA_DIR)
    ap.add_argument("--patients", nargs="+", help="default: every patient folder in --data-root")
    ap.add_argument("--views-per-patient", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "ridge_value.csv"))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    patients = args.patients or sorted(p for p in os.listdir(args.data_root) if p.isdigit()
                                       and os.path.exists(os.path.join(args.data_root, p, "syn", "camPose.yml")))

    rows = list(csv.DictReader(open(args.out))) if os.path.exists(args.out) else []
    if not args.summary_only:
        done = {(r["patient"], int(r["view"]), r["condition"]) for r in rows}
        tasks = []
        for pid in patients:
            P = patient(args.data_root, pid)  # also computes the ridge once, before the workers start
            for view_id, _, _, T in training_views(P, np.random.default_rng([1, int(pid)]), args.views_per_patient):
                todo = [c for c in CONDITIONS if (pid, view_id, c) not in done]
                if todo:
                    tasks.append((args.data_root, pid, view_id, T, todo))
        print(f"{len(tasks)} views to run on {len(patients)} patients", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        new_file = not os.path.exists(args.out)
        _patients.clear()  # workers start fresh ("spawn"): forking a process with Open3D's threads running can hang
        with open(args.out, "a", newline="") as f, multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for k, out in enumerate(pool.imap_unordered(run_view, tasks), 1):
                w.writerows(out)
                f.flush()
                rows += [{key: str(val) for key, val in r.items()} for r in out]
                print(f"[{k}/{len(tasks)}] " + "  ".join(f"{r['patient']}/{r['view']:05d} {r['condition']} "
                                                          f"{r['error_mm']:.1f} mm {r['failure']}" for r in out), flush=True)
    summary(rows)


if __name__ == "__main__":
    main()
