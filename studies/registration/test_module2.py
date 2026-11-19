"""
Modules 2 + 3 from scratch on clean synthetic training views: no knowledge of the true pose.

For each view: is the chosen answer right (mean error at the 200 probe tumours < 5 mm)? If not,
was a right answer among the fully refined candidates (picked wrong) or not (right start missing)?
Uses the same training views as capture_range.py.

Usage: python registration/test_module2.py [--views-per-patient 10] [--start-size 1.25]
"""
import argparse
import csv
import os
import time

import numpy as np

from capture_range import HERE, N_PROBES, PATIENTS, SUCCESS_MM, load_patient, sample_tumours, training_views
from multistart import register_from_scratch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-per-patient", type=int, default=10)
    ap.add_argument("--start-size", type=float, default=1.25)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for pid in PATIENTS:
        P = load_patient(pid)
        probes, _ = sample_tumours(P, np.random.default_rng([0, int(pid)]), N_PROBES)
        for view, cloud, rays, T in training_views(P, np.random.default_rng([1, int(pid)]), args.views_per_patient):
            Q_true, d_true = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
            cam_true = (probes - d_true) @ Q_true
            tre = lambda p: np.linalg.norm((probes - p[2]) @ p[1] - cam_true, axis=1).mean()  # noqa: E731

            t0 = time.time()
            best, finals, errors = register_from_scratch(cloud, rays, P["scene"], P["centroid"], args.start_size)
            rows.append(dict(patient=pid, view=view, tre_mm=tre(best), size=best[0], fit_mm=min(errors),
                             right_among_candidates=min(tre(p) for p in finals) < SUCCESS_MM,
                             seconds=time.time() - t0))
            r = rows[-1]
            print(f"[{pid}] view {view:05d}: tumour error {r['tre_mm']:6.1f} mm, size x{r['size']:.3f}, "
                  f"right answer among candidates: {r['right_among_candidates']}, {r['seconds']:.0f} s", flush=True)

    with open(os.path.join(args.out, f"module2_start{args.start_size}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = np.array([r["tre_mm"] < SUCCESS_MM for r in rows])
    among = np.array([r["right_among_candidates"] for r in rows])
    print(f"\n{len(rows)} views: right {100 * ok.mean():.0f} %, picked wrong {100 * (~ok & among).mean():.0f} %, "
          f"right start missing {100 * (~ok & ~among).mean():.0f} %; "
          f"median {np.median([r['seconds'] for r in rows]):.0f} s per view")


if __name__ == "__main__":
    main()
