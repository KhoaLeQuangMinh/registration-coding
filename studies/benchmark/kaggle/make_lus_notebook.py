"""
Build selfp2ir_lus_kaggle.ipynb: the authors' pretrained Self-P2IR on all 46 LLR-LUS images, on Kaggle,
plus a check of whether Self-P2IR cares how the CT is placed in its own coordinates.

Inputs on Kaggle: the P2I-LReg dataset (for the frame check) and the PRIVATE dataset made from
dataset/lus-evaluation/kaggle_upload/lus-selfp2ir.zip (lus_kaggle_pack.py pack).
The notebook writes our scripts into the session itself, so rerun this after changing any of them:
    python studies/benchmark/kaggle/make_lus_notebook.py            -> selfp2ir_lus_kaggle.ipynb
    python studies/benchmark/kaggle/make_lus_notebook.py --oracle   -> selfp2ir_lus_oracle_kaggle.ipynb
The --oracle notebook runs Self-P2IR once on clouds put at the TRUE distance (lus_oracle.py's oracle.json,
embedded in the notebook; the LUS upload is the same) -- an upper bound, not a method.
"""
import json
import os
import sys

ORACLE = "--oracle" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
EMBED = ["scale_recipe.py", "selfp2ir_predict.py", "selfp2ir_real.py", "lus_kaggle_pack.py",
         "tools/build_cpp_wrappers.py", "tools/pointnet2_torch.py"]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


if ORACLE:
    md("""
# Oracle test: Self-P2IR on LLR-LUS with every cloud at the TRUE distance

Same model, photos, masks and Depth Anything shape as the main LUS run. Only the cloud's scale (sigma) differs:
it is set so the cloud's depth next to the ultrasound probe equals the probe's true contact depth (from the
probe tracking, which also produces the ground truth). This measures how good Self-P2IR could be if the
distance were right: an UPPER BOUND, never a method.

**Before running**
1. *Add Input*: your private LUS dataset (the corrected `lus-selfp2ir.zip`, the same one as the main run).
2. *Settings*: **Internet on**; accelerator **GPU T4**.
3. *Save Version -> Save & Run All*. Building the renderer (PyTorch3D) takes about 15 minutes.
4. Download `selfp2ir_lus_oracle_results.zip` and unzip it into `dataset/lus-evaluation/selfp2ir_results/`.
""")
else:
    md("""
# Self-P2IR on LLR-LUS: Dice and tumour error on the same predictions

1. Self-P2IR (released weights, run as the authors' real-frame tester) on all 46 LLR-LUS images. The LUS clouds
   were made with the recipe recovered from the released P2I-LReg clouds, from hand-corrected liver masks.
   Saves per image the pose, Dice, the bent CT and the cloud, for viewing and scoring on the laptop.
2. Frame check: does Self-P2IR's accuracy change when the CT is rotated or moved in its own coordinates?
   (LLR-LUS CTs are rotated ~45 deg and placed up to 2 m away compared with P2I-LReg's.) 100 synthetic
   P2I-LReg views of patient 01, where the true pose is known.

**Before running**
1. *Add Input*: the dataset `junezhou001/p2i-lreg`, and your private LUS dataset (the uploaded `lus-selfp2ir.zip`).
2. *Settings*: **Internet on**; accelerator **GPU T4**.
3. *Save Version -> Save & Run All*. Building the renderer (PyTorch3D) takes about 15 minutes.
4. Download `selfp2ir_lus_results.zip` and unzip it into `dataset/lus-evaluation/selfp2ir_results/`.
""")

md("## 1. Environment")
code("""
!nvidia-smi -L
!pip install -q open3d easydict nibabel iopath fvcore
import sys, torch, numpy, open3d
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "numpy", numpy.__version__, "open3d", open3d.__version__)
assert torch.cuda.is_available(), "turn the GPU on: Settings -> Accelerator -> GPU T4"
""")
code("""
# PyTorch3D, built from source at a pinned commit. A recent PyTorch needs C++20 where its setup.py
# asks for C++17, so if the first build fails the flag is changed and the build repeated.
P3D = "/kaggle/working/src_pytorch3d"
P3D_COMMIT = "fdd657494ea5151cd66e19a1594494f1e1d592a1"
BUILD = "FORCE_CUDA=1 pip install -q --no-build-isolation --no-deps ."
try:
    import pytorch3d
except ImportError:
    !git clone -q https://github.com/facebookresearch/pytorch3d.git {P3D} && git -C {P3D} checkout -q {P3D_COMMIT}
    !cd {P3D} && ({BUILD} > /kaggle/working/p3d_build.log 2>&1 || (echo "retrying with C++20" && sed -i 's/-std=c++17/-std=c++20/g' setup.py && rm -rf build && {BUILD} 2>&1 | tail -5))
    import pytorch3d
from pytorch3d.ops import knn_points
from pytorch3d.renderer import PointsRasterizer
print("pytorch3d", pytorch3d.__version__)
""")

md("## 2. Find the two datasets")
code(f"ORACLE = {ORACLE}")
code("""
import glob, os
hits = sorted(set(os.path.dirname(os.path.dirname(os.path.dirname(p)))
                  for depth in range(1, 7)
                  for p in glob.glob("/kaggle/input/" + "*/" * depth + "model/reconstructed_mesh_world_m.obj")
                  if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), "syn", "camPose.yml"))))
P2I = hits[0] if len(hits) == 1 else None
assert P2I or ORACLE, f"expected one P2I-LReg root, found {hits}"
lus = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/meta.json", recursive=True)
             if os.path.exists(os.path.join(os.path.dirname(p), "01", "real", "CAM_K.yml")))
assert len(lus) == 1, f"expected one LUS upload (a folder with meta.json and 01/real/CAM_K.yml), found {lus}"
LUS_UPLOAD = lus[0]
print("P2I-LReg:", P2I)
print("LUS upload:", LUS_UPLOAD)
""")

md("## 3. Self-P2IR at the pinned commit, and its pretrained weights")
code("""
COMMIT = "60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079"
WEIGHTS_SHA256 = "e83a9110bb6a1b1bf7e49e7e0f3a3736b0b01ce9ec76626972a5406afaa963ba"
REPO = "/kaggle/working/Self-P2IR"
WEIGHTS = "/kaggle/working/model_best_loss.pth"
if not os.path.exists(REPO):
    !git clone -q https://github.com/junzastar/Self-P2IR.git {REPO}
!git -C {REPO} checkout -q {COMMIT} && git -C {REPO} rev-parse HEAD
if not os.path.exists(WEIGHTS):
    !curl -sL -o {WEIGHTS} "https://drive.usercontent.google.com/download?id=1xz1u4jtc-J3g8IpgbjHMBS9CVpCCK4nO&export=download&confirm=t"
import hashlib
digest = hashlib.sha256(open(WEIGHTS, "rb").read()).hexdigest()
assert digest == WEIGHTS_SHA256, f"weights checksum {digest} differs from the expected one"
print("weights checksum ok")
""")

md("## 4. Our scripts (embedded from the repo by make_lus_notebook.py)")
code("os.makedirs('/kaggle/working/benchmark/tools', exist_ok=True)")
for name in EMBED:
    code(f"%%writefile /kaggle/working/benchmark/{name}\n" + open(os.path.join(BENCH, name)).read())
if ORACLE:
    md("The oracle scales (lus_oracle.py), one per image:")
    code("%%writefile /kaggle/working/oracle.json\n" +
         open(os.path.join(BENCH, "..", "..", "dataset", "lus-evaluation", "selfp2ir", "oracle.json")).read())

md("## 5. Build the two C++ helpers, and rebuild the LUS clouds from their depth maps")
code("""
%cd /kaggle/working
!python benchmark/tools/build_cpp_wrappers.py {REPO} /kaggle/working/build 2>&1 | grep -E "^built| error"
LUS = "/kaggle/working/lus_selfp2ir"
!python -W ignore benchmark/lus_kaggle_pack.py unpack {LUS_UPLOAD} {LUS} {"/kaggle/working/oracle.json" if ORACLE else ""}
""")

if ORACLE:
    md("""
## 6. Self-P2IR on the 46 LUS images, clouds at the true distance
LUS has no training frames, so the neighbourhood limits are the ones calibrated on P2I-LReg's real training
frames (all 21 patients): 9 19 25 30. CT livers as given (centring them did not help). One line per image.
""")
else:
    md("""
## 6. Self-P2IR on the 46 LUS images, twice
LUS has no training frames, so the neighbourhood limits are the ones calibrated on P2I-LReg's real training
frames (all 21 patients): 9 19 25 30. One line per image.

- **as given**: the LUS CT livers in their own coordinates (rotated ~45 deg and up to 2 m from the origin);
- **centred**: the same CTs moved so their mean is at the origin, as P2I-LReg's are. The first frame check
  showed Self-P2IR loses accuracy when the CT is moved, so this is the fair placement. Clouds are identical.
""")
code("""
# Sanity check that this is the corrected upload (2026-11-13): the cloud must be scaled about its bounding-box
# centre, which leaves the PLY's bounding-box centre off zero (the first, wrong upload centred it exactly).
import numpy as np
P = np.asarray(open3d.io.read_point_cloud(f"{LUS}/01/real/liverPcds/frame_2_json.ply").points)
c = (P.max(0) + P.min(0)) / 2
print("bounding-box centre of 01/frame_2 (mm):", np.round(c, 3))
assert np.abs(c).max() > 0.05, "this is the OLD upload: upload the corrected lus-selfp2ir.zip as a new dataset version"
""")
code("""
if ORACLE:
    RUNS_LUS = {"lus_oracle": LUS}
else:
    RUNS_LUS = {"lus_results": LUS, "lus_results_centred": "/kaggle/working/lus_selfp2ir_centred"}
    !python -W ignore benchmark/lus_kaggle_pack.py centre-ct {LUS} {RUNS_LUS["lus_results_centred"]}
for out, root in RUNS_LUS.items():
    print("=====", out)
    OUT = f"/kaggle/working/{out}"
    !python -W ignore benchmark/selfp2ir_real.py --repo {REPO} --weights {WEIGHTS} --data-root {root} --patients 01 02 04 05 --limits 9 19 25 30 --out {OUT} --save-points {OUT}/points 2>&1 | grep -v "buffer loaded"
    !cp {root}/meta.json {OUT}/
    if os.path.exists(f"{root}/ct_offsets.json"):
        !cp {root}/ct_offsets.json {OUT}/
""")

if not ORACLE:
    md("""
## 7. Frame check: does moving the CT change Self-P2IR's answer?
Already run once (2026-11-13); set RUN_FRAME_CHECK = True to repeat it.""")
    code("RUN_FRAME_CHECK = False")
    md("""
### 7a. The check
Same 100 synthetic views of P2I-LReg patient 01, four times: CT as given; rotated 45 deg about z; moved 0.25 m;
moved 2 m (LUS Patient 3's CT sits ~2 m from the origin). Registration recall at 5 mm, as in the paper.
""")
    code("""
FC = "/kaggle/working/frame_check"
os.makedirs(FC, exist_ok=True)
RUNS = {"as given": "", "rotated 45 deg": "--ct-rot-z 45", "moved 0.25 m": "--ct-shift 0 0 0.25", "moved 2 m": "--ct-shift 0 0 2.0"}
for name, extra in (RUNS.items() if RUN_FRAME_CHECK else []):
    print("=====", name)
    !python -W ignore benchmark/selfp2ir_predict.py --repo {REPO} --weights {WEIGHTS} --data-root {P2I} --condition A --patients 01 --limit 100 {extra} --out {FC} 2>&1 | grep -E "saved|rror"
""")
    code("""
import json, re, numpy as np
V = np.asarray(open3d.io.read_triangle_mesh(f"{P2I}/01/model/reconstructed_mesh_world_m.obj").vertices)  # metres
print(f"{'CT placement':38s} views  recall<5mm  median error")
for f in sorted(glob.glob(f"{FC}/predictions_*.json")):
    tag = os.path.basename(f)[len("predictions_"):-5]
    rot = float(re.search(r"ctrot([-\\d.]+)", tag).group(1)) if "ctrot" in tag else 0.0
    shift = np.array(re.search(r"ctshift(.+)$", tag).group(1).split("_"), float) if "ctshift" in tag else np.zeros(3)
    a = np.deg2rad(rot)
    G = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    X = V @ G.T + shift                                    # the CT as the model saw it
    errs = []
    for r in json.load(open(f))["predictions"]:
        R, t, R0, t0 = (np.array(r[k]) for k in ("R", "t", "R_true", "t_true"))
        errs.append(1000 * np.linalg.norm(X @ (R - R0).T + (t - t0), axis=1).mean())
    errs = np.array(errs)
    print(f"{tag:38s} {len(errs):5d}  {100 * (errs < 5).mean():9.1f} %  {np.median(errs):9.2f} mm")
""")

md("## 8. Download")
if ORACLE:
    code("!cd /kaggle/working && zip -qr selfp2ir_lus_oracle_results.zip lus_oracle && ls -la selfp2ir_lus_oracle_results.zip")
else:
    code("!cd /kaggle/working && zip -qr selfp2ir_lus_results.zip lus_results lus_results_centred frame_check && ls -la selfp2ir_lus_results.zip")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
name = "selfp2ir_lus_oracle_kaggle.ipynb" if ORACLE else "selfp2ir_lus_kaggle.ipynb"
with open(os.path.join(HERE, name), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.join(HERE, name), f"({len(cells)} cells)")
