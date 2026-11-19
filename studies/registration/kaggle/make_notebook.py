"""
Build ridge_value_kaggle.ipynb: registration/test_ridge_value.py on all P2I-LReg patients, on Kaggle.

The notebook writes our scripts into the session itself (no git push needed), so rerun this
after changing any of them:  python registration/kaggle/make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EMBED = ["benchmark/pilot_dice_vs_tumour.py", "registration/sim3_icp.py", "registration/capture_range.py",
         "registration/multistart.py", "registration/outline_reg.py", "registration/ct_landmarks.py",
         "registration/test_ridge_value.py"]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# Does a ridge help mask-only registration beyond searching harder? All P2I-LReg patients

Design, conditions and the decision rule: the docstring of `registration/test_ridge_value.py` (written below).
Three conditions per view: `outline` (mask only), `ridge` (mask + a perfect ridge), `outline_more`
(mask only, 4x the search). Primary result: ridge - outline_more on the 17 patients not used in development.

**Before running**
1. *Add Input*: the dataset `junezhou001/p2i-lreg`.
2. *Settings*: **Internet on** (for `pip install open3d`); accelerator **None** (CPU only; a GPU does not help).
3. *Save Version -> Save & Run All*. The run stops itself after `HOURS` and keeps every finished view.
   If the summary says views are missing, add this version's output as an input (*Add Input* -> *Notebook Output*)
   and run a new version: finished views are copied in and not recomputed.
""")

code("""
HOURS = 11                 # Kaggle stops a run at 12 h; stop ours earlier so the output is saved
VIEWS_PER_PATIENT = 10     # fixed in test_ridge_value.py's docstring
""")

md("## 1. Environment")
code("""
!pip install -q open3d
import os, sys, numpy, scipy, open3d
WORKERS = os.cpu_count()
print(sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__,
      "open3d", open3d.__version__, "cpus", WORKERS)
""")

md("## 2. Find the dataset and the patients")
code("""
import glob
# e.g. /kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis/01/model/reconstructed_mesh_world_m.obj
hits = sorted(set(os.path.dirname(os.path.dirname(os.path.dirname(p)))
                  for depth in range(1, 7)
                  for p in glob.glob("/kaggle/input/" + "*/" * depth + "model/reconstructed_mesh_world_m.obj")))
assert len(hits) == 1, f"expected one dataset root, found {hits}"
ROOT = hits[0]
PATIENTS = sorted(p for p in os.listdir(ROOT) if p.isdigit() and os.path.exists(f"{ROOT}/{p}/syn/camPose.yml"))
DEV = ["01", "06", "10", "11"]
ORDER = [p for p in PATIENTS if p not in DEV] + [p for p in PATIENTS if p in DEV]  # confirmatory patients first
print("dataset root:", ROOT)
print(len(PATIENTS), "patients, run in this order:", " ".join(ORDER))
""")

md("## 3. Our scripts (embedded from the repo by make_notebook.py)")
code("os.makedirs('/kaggle/working/benchmark', exist_ok=True); os.makedirs('/kaggle/working/registration', exist_ok=True)")
for name in EMBED:
    code(f"%%writefile /kaggle/working/{name}\n" + open(os.path.join(REPO, name)).read())

md("## 4. Finished views from earlier versions")
code("""
import shutil
OUT = "/kaggle/working/ridge_out"
CSV = f"{OUT}/ridge_value.csv"
os.makedirs(OUT, exist_ok=True)
os.makedirs("/kaggle/working/registration/landmarks", exist_ok=True)
for prev in glob.glob("/kaggle/input/" + "**/ridge_out/ridge_value.csv", recursive=True):
    if not os.path.exists(CSV):
        shutil.copy2(prev, CSV)
        print("copied", prev)
for prev in glob.glob("/kaggle/input/" + "**/registration/landmarks/*.json", recursive=True):
    dst = os.path.join("/kaggle/working/registration/landmarks", os.path.basename(prev))
    if not os.path.exists(dst):
        shutil.copy2(prev, dst)
print(sum(1 for _ in open(CSV)) - 1 if os.path.exists(CSV) else 0, "rows already done")
""")

md("""
## 5. Run
The ridge of each CT is computed once with the frozen rule (`ct_landmarks.py`) and saved in `registration/landmarks/`.
""")
code("""
%cd /kaggle/working
cmd = (f"PYTHONPATH=registration:benchmark timeout {HOURS}h python -W ignore registration/test_ridge_value.py "
       f"--data-root {ROOT} --patients {' '.join(ORDER)} --views-per-patient {VIEWS_PER_PATIENT} "
       f"--workers {WORKERS} --out {CSV}")
!{cmd}; echo "exit code $? (124: stopped after {HOURS} h, run a new version to finish)"
""")

md("## 6. Summary (finished views only)")
code("""
!PYTHONPATH=registration:benchmark python -W ignore registration/test_ridge_value.py --data-root {ROOT} --out {CSV} --summary-only
""")

md("## 7. Download")
code("""
!cd /kaggle/working && zip -qr ridge_value_results.zip ridge_out registration && ls -la ridge_value_results.zip
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(HERE, "ridge_value_kaggle.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.join(HERE, "ridge_value_kaggle.ipynb"), f"({len(cells)} cells)")
