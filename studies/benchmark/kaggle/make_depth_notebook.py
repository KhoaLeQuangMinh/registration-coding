"""
Build depth_benchmark_kaggle.ipynb: which single-image depth model gets the distance right in laparoscopy
(route 1, step 1), on Kaggle, where the model weights are downloaded (not on the laptop).

Input on Kaggle: the PRIVATE LLR-LUS dataset (the corrected lus-selfp2ir.zip). The probe contact depths and regions
(oracle.json) and depth_benchmark.py are written into the notebook. Rebuild after changing either:
    python studies/benchmark/kaggle/make_depth_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BENCH))

# model -> (title, install commands); the installs most likely to clash with Kaggle's PyTorch come last
MODELS = [
    ("dav1indoor", "Depth Anything V1 metric indoor: Self-P2IR's own depth, before its scale step (no download)", ""),
    ("dav2indoor", "Depth Anything V2 Metric-Indoor-Large (transformers)", ""),
    # --no-deps keeps Kaggle's PyTorch (the repos pin their own); their other requirements are listed by hand
    ("depthpro", "Apple Depth Pro",
     "pip install -q --no-deps git+https://github.com/apple/ml-depth-pro.git && pip install -q timm pillow_heif"),
    ("metric3d", "Metric3D v2 ViT-L (torch.hub)", "pip install -q mmengine timm"),
    ("unidepth", "UniDepthV2 ViT-L",
     "pip install -q --no-deps git+https://github.com/lpiccinelli-eth/UniDepth.git && "
     "pip install -q einops timm h5py tables tabulate termcolor trimesh wandb"),
    ("moge", "MoGe-2 ViT-L", "pip install -q git+https://github.com/microsoft/MoGe.git"),
    ("da3metric", "Depth Anything 3 METRIC-LARGE",
     # Kaggle's Python 3.13.16 fails the repo's "<=3.13" pin on a technicality
     "pip install -q --no-deps --ignore-requires-python git+https://github.com/ByteDance-Seed/Depth-Anything-3.git && "
     "pip install -q einops omegaconf addict safetensors trimesh imageio evo e3nn 'moviepy==1.0.3' plyfile pillow_heif "
     "pycolmap typer"),
    ("da3nested", "Depth Anything 3 NESTED-GIANT-LARGE-1.1 (outputs metres)", ""),
]

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n")))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n")))


md("""
# Which depth model gets the distance right in laparoscopy?

On LLR-LUS the tracked ultrasound probe touches the liver, so the true depth of the liver surface there is known
(46 images). Each model predicts metric depth for the photo; we compare its depth on the visible liver around the
probe tip with the true contact depth. The probe is used to **evaluate** only, never as an input.

Each model runs in its own cell and its own process: if one fails to install or load, the others still run.
Its weights are downloaded here and deleted after its run.

**Before running**
1. *Add Input*: your private LUS dataset (the corrected `lus-selfp2ir.zip`).
2. *Settings*: **Internet on**; accelerator **GPU T4**.
3. *Save Version -> Save & Run All* (roughly 30-60 minutes, mostly downloads).
4. Download `depth_benchmark_results.zip` (small: numbers and preview images). If a model failed, paste its cell's log.
""")

md("## 1. Environment and the LUS upload")
code("""
!nvidia-smi -L
import glob, os, sys, torch
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "turn the GPU on: Settings -> Accelerator -> GPU T4"
lus = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/meta.json", recursive=True)
             if os.path.exists(os.path.join(os.path.dirname(p), "01", "real", "CAM_K.yml")))
assert len(lus) == 1, f"expected one LUS upload (a folder with meta.json and 01/real/CAM_K.yml), found {lus}"
LUS = lus[0]
print("LUS upload:", LUS)
os.makedirs("/kaggle/working/bench", exist_ok=True)
%cd /kaggle/working/bench
""")

md("## 2. The benchmark script and the probe contact depths (embedded by make_depth_notebook.py)")
code("%%writefile /kaggle/working/bench/depth_benchmark.py\n" + open(os.path.join(BENCH, "depth_benchmark.py")).read())
code("%%writefile /kaggle/working/bench/oracle.json\n" +
     open(os.path.join(REPO, "dataset", "lus-evaluation", "selfp2ir", "oracle.json")).read())

md("## 3. One model per cell")
code("""
OUT = "/kaggle/working/depth_out"
def run_model(name):
    !python -W ignore depth_benchmark.py --lus {LUS} --oracle oracle.json --model {name} --out {OUT} 2>&1 | grep -v "^\\s*$" | tail -60
    !rm -rf ~/.cache/huggingface/hub ~/.cache/torch/hub/checkpoints checkpoints   # free the disk for the next model
""")
for name, title, install in MODELS:
    md(f"### {title}")
    code((f"!{install}\n" if install else "") + f'run_model("{name}")')

md("## 4. Summary: every model that ran, against the true contact depth")
code("""
!python -W ignore depth_benchmark.py --lus {LUS} --oracle oracle.json --summary --out {OUT}
""")

md("## 5. Download")
code("""
!cd /kaggle/working && zip -qr depth_benchmark_results.zip depth_out && ls -la depth_benchmark_results.zip
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(HERE, "depth_benchmark_kaggle.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.join(HERE, "depth_benchmark_kaggle.ipynb"), f"({len(cells)} cells)")
