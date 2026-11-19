"""Plots + numbers for goal/out/landscape.json (goal_landscape.py). Needs matplotlib.
Usage: python goal/plot_landscape.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "goal/out"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

d = json.load(open(f"{OUT}/landscape.json"))
G = d["goals"]
X = {k: np.array(v) for k, v in d["sweeps"].items()}
TRUE = {k: (1.0 if k == "distance" else 0.0) for k in X}
UNIT = {"distance": "CT distance (× true)", "side_x": "sideways, camera x (mm)", "side_y": "sideways, camera y (mm)",
        "rot_x": "rotation about camera x (deg)", "rot_y": "rotation about camera y (deg)", "rot_z": "rotation about view axis (deg)"}


def stack(cond, sweep):
    """views x steps x goals"""
    return np.array([v[cond][sweep] for p in d["patients"].values() for v in p["phase2"] if v.get(cond)])


def band(ax, x, Y, color, label, ls="-"):
    q1, med, q3 = np.percentile(Y, [25, 50, 75], axis=0)
    ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)
    ax.plot(x, med, color=color, lw=2, ls=ls, label=label)


def panels(cond, sweeps, fname, title):
    fig, axs = plt.subplots(len(sweeps), 3, figsize=(10, 2.5 * len(sweeps)), squeeze=False)
    for r, s in enumerate(sweeps):
        Y, x = stack(cond, s), X[s]
        a = axs[r]
        band(a[0], x, Y[:, :, G.index("dice_plain")], BLUE, "1 − Dice, real camera")
        band(a[0], x, Y[:, :, G.index("dice_masked")], ORANGE, "1 − Dice, masked", ls="--")
        if s == "distance":
            a[0].plot(x, np.zeros_like(x), color=MUTED, lw=1.5, ls=":", label="size free with distance\n(Self-P2IR-style; flat by construction)")
        band(a[1], x, Y[:, :, G.index("A")], BLUE, "A1 + A2")
        band(a[2], x, Y[:, :, G.index("B")], BLUE, "B")
        a[0].set_ylabel(UNIT[s].split(" (")[0] + "\n\n1 − Dice")
        a[1].set_ylabel("A1 + A2 (px, 1920-wide)")
        a[2].set_ylabel("B: 1 − cos(normals)")
        for ax in a:
            ax.axvline(TRUE[s], color=INK, lw=0.8)
            ax.set_xlabel(UNIT[s])
        if r == 0:
            a[0].legend(frameon=False, fontsize=7.5, loc="upper center")
            a[0].set_title("Overlap goals", loc="left", fontsize=10, color=INK)
            a[1].set_title("Outline distance goal", loc="left", fontsize=10, color=INK)
            a[2].set_title("Shape goal (scale-free)", loc="left", fontsize=10, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Line: median of 80 views (4 patients x 20); band: middle 50 %. Vertical line: the true pose. Lower = better.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(f"{OUT}/{fname}", dpi=130)
    plt.close(fig)


def summary(cond):
    rows = []
    for s, x in X.items():
        Y = stack(cond, s)
        c = int(np.argmin(np.abs(x - TRUE[s])))
        for g in ["dice_plain", "dice_masked", "A", "B"]:
            y = Y[:, :, G.index(g)]
            at_truth = np.mean(np.argmin(y, 1) == c)
            # basin: how far from the truth the goal keeps rising on BOTH sides (no dip, ties count as a stop)
            ext = []
            for v in y:
                L = 0
                while c - L - 1 >= 0 and v[c - L - 1] > v[c - L]:
                    L += 1
                Rr = 0
                while c + Rr + 1 < len(v) and v[c + Rr + 1] > v[c + Rr]:
                    Rr += 1
                ext.append(min(L, Rr))
            step = x[1] - x[0]
            rows.append((s, g, at_truth, np.median(ext) * step, np.percentile(ext, 10) * step))
    return rows


for cond in ["clean", "scope"]:
    panels(cond, ["distance"], f"distance_{cond}.png",
           f"Moving the CT nearer / farther ({'only the image border hides anything' if cond == 'clean' else 'round scope border hides the corners'})")
panels("clean", ["side_x", "side_y", "rot_x", "rot_y", "rot_z"], "other_clean.png", "Moving the CT sideways and rotating it (clean)")

lines = []
for cond in ["clean", "scope"]:
    n = len(stack(cond, "distance"))
    lines.append(f"\n== {cond} ({n} views) ==  goal | best score at the truth | basin half-width median (10th pct)")
    for s, g, at, med, p10 in summary(cond):
        u = "x" if s == "distance" else ("mm" if s.startswith("side") else "deg")
        lines.append(f"  {s:8s} {g:11s}  {at:5.0%}   ±{med:5.2f} {u} ({p10:.2f})")

# distance steepness: score at x0.9 / x1.1, clean
Y = stack("clean", "distance"); x = X["distance"]
for g in ["dice_plain", "A", "B"]:
    i9, i11 = np.argmin(np.abs(x - 0.9)), np.argmin(np.abs(x - 1.1))
    lines.append(f"  distance x0.9 / x1.1, {g}: median {np.median(Y[:, i9, G.index(g)]):.3f} / {np.median(Y[:, i11, G.index(g)]):.3f}")

# phase 3: wrong poses all around
lines.append("\n== all-around scan (clean): best A1+A2 near the truth (<20 deg) vs far from it (>45 deg) ==")
fig, ax = plt.subplots(figsize=(6, 3.6))
for p, pv in d["patients"].items():
    for v in pv.get("phase3", []):
        ang = np.array([r["angle_to_true"] for r in v["directions"]]); A = np.array([r["A"] for r in v["directions"]])
        B = np.array([r["B"] for r in v["directions"]])
        near, far = A[ang < 20].min() if (ang < 20).any() else np.nan, A[ang > 45].min()
        jf = np.argmin(np.where(ang > 45, A, np.inf))
        lines.append(f"  {p}/{v['view']}: near {near:6.1f} px, best far {far:6.1f} px at {ang[jf]:5.0f} deg (B there {B[jf]:.2f}; "
                     f"B near {B[np.argmin(np.where(ang < 20, A, np.inf))]:.2f})  -> {'FALSE VALLEY' if far <= near else 'ok'}")
        ax.scatter(ang, A, s=10, color=BLUE, alpha=0.45, lw=0)
ax.set_xlabel("camera direction: angle from the true one (deg)")
ax.set_ylabel("best A1 + A2 (px)")
ax.set_title("Cameras all around the liver: best outline score per direction (12 views)", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/allaround.png", dpi=130)
open(f"{OUT}/summary.txt", "w").write("\n".join(lines))
print("\n".join(lines))
