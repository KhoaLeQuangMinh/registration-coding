"""
Build selfp2ir_real_kaggle.ipynb: the authors' pretrained Self-P2IR on P2I-LReg real test frames, on Kaggle.

The notebook writes our scripts into the session itself (no git push needed), so rerun this
after changing any of them:  python studies/benchmark/kaggle/make_real_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
EMBED = ["scale_recipe.py", "selfp2ir_predict.py", "selfp2ir_real.py",
         "tools/build_cpp_wrappers.py", "tools/pointnet2_torch.py"]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# Do the released Self-P2IR weights reproduce the paper on real frames?

Runs the authors' pretrained model on every real test frame of P2I-LReg (`test_real.txt` of each patient),
following their own real tester, and prints their two numbers.
Paper (Table II, mean of five folds): **Dice 78.89 +- 6.76 %, CD 2.97 +- 1.06**. The released weights are one fold.

**Before running**
1. *Add Input*: the dataset `junezhou001/p2i-lreg`.
2. *Settings*: **Internet on**; accelerator **GPU T4** (the model's renderer needs a GPU).
3. *Save Version -> Save & Run All*. Building the renderer (PyTorch3D) takes about 15 minutes.

Checked on a laptop CPU first (patient 01, first two frames): Dice 81.5 and 79.1.
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

md("## 2. Find the dataset and the patients")
code("""
import glob, os
hits = sorted(set(os.path.dirname(os.path.dirname(os.path.dirname(p)))
                  for depth in range(1, 7)
                  for p in glob.glob("/kaggle/input/" + "*/" * depth + "model/reconstructed_mesh_world_m.obj")))
assert len(hits) == 1, f"expected one dataset root, found {hits}"
ROOT = hits[0]
PATIENTS = sorted(p for p in os.listdir(ROOT) if p.isdigit() and os.path.exists(f"{ROOT}/{p}/test_real.txt"))
print("dataset root:", ROOT)
for p in PATIENTS:
    n = [sum(1 for line in open(f"{ROOT}/{p}/{name}_real.txt") if line.strip()) for name in ("train", "test")]
    print(f"patient {p}: {n[0]:3d} training frames, {n[1]:3d} test frames")
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

md("## 4. Our scripts (embedded from the repo by make_real_notebook.py)")
code("os.makedirs('/kaggle/working/benchmark/tools', exist_ok=True)")
for name in EMBED:
    code(f"%%writefile /kaggle/working/benchmark/{name}\n" + open(os.path.join(BENCH, name)).read())

md("## 5. Build the two C++ helpers")
code("""
%cd /kaggle/working
!python benchmark/tools/build_cpp_wrappers.py {REPO} /kaggle/working/build 2>&1 | grep -E "^built| error"
""")

md("""
## 6. Run
One line per frame, then the mean Dice and CD next to the paper's.
""")
code("""
OUT = "/kaggle/working/selfp2ir_real_out"
!python -W ignore benchmark/selfp2ir_real.py --repo {REPO} --weights {WEIGHTS} --data-root {ROOT} --out {OUT} 2>&1 | grep -v "buffer loaded"
""")

md("## 7. Download")
code("""
!cd /kaggle/working && zip -qr selfp2ir_real_results.zip selfp2ir_real_out && ls -la selfp2ir_real_results.zip
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(HERE, "selfp2ir_real_kaggle.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.join(HERE, "selfp2ir_real_kaggle.ipynb"), f"({len(cells)} cells)")
