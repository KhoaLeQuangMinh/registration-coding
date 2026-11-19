"""
Build selfp2ir_kaggle.ipynb: the Self-P2IR study on all P2I-LReg patients, on Kaggle.

The notebook writes our scripts into the session itself (no git push needed), so rerun this
after changing any of them:  python benchmark/kaggle/make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
EMBED = ["pilot_dice_vs_tumour.py", "scale_recipe.py", "selfp2ir_predict.py", "score_selfp2ir.py",
         "tools/build_cpp_wrappers.py", "PREREGISTRATION.md"]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# Does Dice tell us where the tumour is? Self-P2IR on all P2I-LReg patients

Pre-registered study: see `PREREGISTRATION.md` (written into `benchmark/` below). This notebook runs
the authors' pretrained Self-P2IR on every synthetic test view, then scores H1-H3 and E1.

**Before running**
1. *Add Input*: the dataset `junezhou001/p2i-lreg`.
2. *Settings*: **Internet on** (needs a phone-verified Kaggle account); accelerator GPU (T4 or P100) if available, CPU also works.
3. Kaggle keeps `/kaggle/working` only from a **Save Version -> Save & Run All** run, and a run stops after 12 hours.
   So split the runs over versions with `RUN_NOW` below, and pass each finished version's output to the next one
   (*Add Input* -> *Notebook Output*). Predictions found there are copied in and not recomputed.
""")

code("""
# ---- What this version runs -------------------------------------------------------------------
RUNS = {                                   # name -> arguments, fixed in PREREGISTRATION.md
    "A":        "--condition A",
    "B_recipe": "--condition B",
    "B_x1.05":  "--condition B --factor 1.05",
    "B_x1.11":  "--condition B --factor 1.11",
    "B_x1.32":  "--condition B --factor 1.32",
}
RUN_NOW = ["A", "B_recipe"]               # e.g. version 1: A + B_recipe, version 2: the three fixed factors
SCORE = False                             # True in the last version, once all five runs exist
""")

md("## 1. Environment")
code("""
!nvidia-smi -L || echo "no GPU: running on CPU"
!pip install -q open3d easydict nibabel
import sys, torch, numpy, open3d
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "numpy", numpy.__version__, "open3d", open3d.__version__)
""")

md("## 2. Find the dataset and the patients")
code("""
import glob, os
# e.g. /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis/01/model/reconstructed_mesh_world_m.obj
hits = sorted(set(os.path.dirname(os.path.dirname(os.path.dirname(p)))
                  for depth in range(1, 7)
                  for p in glob.glob("/kaggle/input/" + "*/" * depth + "model/reconstructed_mesh_world_m.obj")))
assert len(hits) == 1, f"expected one dataset root, found {hits}"
ROOT = hits[0]
PATIENTS = sorted(p for p in os.listdir(ROOT) if p.isdigit() and
                  os.path.exists(f"{ROOT}/{p}/test_syn.txt") and os.path.exists(f"{ROOT}/{p}/syn/camPose.yml"))
print("dataset root:", ROOT)
print(len(PATIENTS), "patients:", " ".join(PATIENTS))
assert PATIENTS[0] == "01", "neighbourhood calibration follows main.py, which starts from patient 01"
PATIENT_ARGS = " ".join(PATIENTS)
""")

md("## 3. Self-P2IR at the pre-registered commit, and its pretrained weights")
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
assert digest == WEIGHTS_SHA256, f"weights checksum {digest} differs from the pre-registered one"
print("weights checksum ok")
""")

md("## 4. Our scripts (embedded from the repo by make_notebook.py)")
code("os.makedirs('/kaggle/working/benchmark/tools', exist_ok=True)")
for name in EMBED:
    code(f"%%writefile /kaggle/working/benchmark/{name}\n" + open(os.path.join(BENCH, name)).read())

md("## 5. Build the two C++ helpers")
code("""
%cd /kaggle/working
!python benchmark/tools/build_cpp_wrappers.py {REPO} /kaggle/working/build 2>&1 | grep -E "^built| error"
""")

md("""
## 6. Predictions
Each run saves every 25 views; if a version stops, pass its output to the next version and the run resumes.
""")
code("""
import shutil
OUT = "/kaggle/working/selfp2ir_out"
os.makedirs(OUT, exist_ok=True)
previous = [p for depth in range(0, 6)                                      # outputs of earlier versions
            for p in glob.glob("/kaggle/input/" + "*/" * depth + "selfp2ir_out/predictions_*.json")]
for prev in previous:
    if not os.path.exists(os.path.join(OUT, os.path.basename(prev))):
        shutil.copy2(prev, OUT)
        print("copied", prev)
print(sorted(os.listdir(OUT)))
""")
code("""
import subprocess

def run(cmd, keep=()):
    \"\"\"Run a command, show the lines containing `keep` (all if empty); on failure show the end and stop.\"\"\"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if not keep or any(k in line for k in keep):
            print(line, end="", flush=True)
    if p.wait() != 0:
        print("".join(tail))
        raise RuntimeError(f"command failed: {cmd}")

for name in RUN_NOW:
    print(f"===== {name} =====")
    run(f"python -W ignore benchmark/selfp2ir_predict.py --repo {REPO} --weights {WEIGHTS} --data-root {ROOT} "
        f"--patients {PATIENT_ARGS} {RUNS[name]} --out {OUT}",
        keep=("neighbourhood", "resuming", "views,", "saved", "complete"))
""")

md("""
## 7. Reproduction check, then the pre-registered analysis
The check uses only the authors' own registration recall (paper: > 92 % at 5 mm). Look at it before scoring.
""")
code("""
if os.path.exists(os.path.join(OUT, "predictions_A.json")):
    run(f"python benchmark/score_selfp2ir.py --pred-dir {OUT} --data-root {ROOT} --check")
else:
    print("no condition A predictions yet: reproduction check skipped")
""")
code("""
if SCORE:
    run(f"python benchmark/score_selfp2ir.py --pred-dir {OUT} --data-root {ROOT} 2>&1 | grep -v 'scored$'")
else:
    print("SCORE is False: set it to True in the version that has all five runs")
""")

md("## 8. Download")
code("""
!cd /kaggle/working && zip -qr selfp2ir_results.zip selfp2ir_out benchmark/PREREGISTRATION.md && ls -la selfp2ir_results.zip
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(HERE, "selfp2ir_kaggle.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.join(HERE, "selfp2ir_kaggle.ipynb"), f"({len(cells)} cells)")
