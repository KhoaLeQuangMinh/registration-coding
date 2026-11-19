"""
Run the authors' pretrained Self-P2IR rigid network on synthetic test views; save the predicted poses.

Follows their synthetic tester (tester.py, _synTester.test_thr) step by step:
  dataset item -> collate_fn_3dmatch -> Pipeline (posenet) -> Matching.get_match(thr=0.15)
  -> MatchMotionLoss.ransac_regist_coarse (weighted SVD) -> R, t  (CT metres -> Blender camera metres)
Neighbourhood limits are calibrated as in main.py: on the augmented training set, whose first
items are patient 01's, with setup_seed(0).

Deviations from their code, all needed to run it here or to keep the test clean:
  - Only the rigid network (posenet.* weights) is built, so pytorch3d and the real-data modules
    are not needed. Weights are loaded with strict=True (their loader uses strict=False).
  - CUDA-only extensions imported by the loss module (Chamfer, EMD) and the plotting module are
    replaced by stubs; none of them is used at test time.
  - Their PoseDataset hard-codes patients 1-21; the list is replaced by --patients.
  - Their top-level folders are registered as packages explicitly, so installed packages with the
    same name (Hugging Face `datasets` on Kaggle) cannot shadow them.
  - A view their loader rejects (< 100 liver pixels or < 1000 correspondences to the TRUE pose)
    is skipped and recorded. Their test loop silently replaces it with a random other view.
  - Without a GPU, tensor.cuda() is a no-op so their solver runs on the CPU.
  - Open3D's random generator (used by their CT point sampling) is seeded, so runs repeat exactly.
Condition B resizes the target cloud after the rejection check, so A and B keep the same views.
Progress is saved every 25 views; rerunning the same command resumes, reusing the saved
neighbourhood limits (Kaggle sessions stop after 12 hours).

Usage:
  python benchmark/selfp2ir_predict.py --repo <Self-P2IR checkout> --weights model_best_loss.pth \
      --condition A [--patients 01 06 10 11] [--out benchmark/selfp2ir_out]
  --condition B             recipe scale (scale_recipe.py, range definition)
  --condition B --factor 1.11  fixed scale factor (sensitivity runs)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
import open3d as o3d
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_recipe import recipe_factor  # noqa: E402

PATIENT_LINE = "self.patients = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]"


def stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod


def import_repo(repo):
    """Make their packages importable without the training-only CUDA extensions."""
    sys.path.insert(0, repo)

    # Their top-level folders have no __init__.py, so an installed package of the same name
    # (e.g. Hugging Face `datasets` on Kaggle) would win; register the repo folders explicitly
    for name in ("configs", "cpp_wrappers", "datasets", "extensions", "lib", "models"):
        sys.modules.pop(name, None)
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(repo, name)]
        sys.modules[name] = pkg

    class TrainingOnly(torch.nn.Module):  # lib/loss.py builds one at import time
        def forward(self, *args, **kwargs):
            raise RuntimeError("training-only CUDA extension, not available here")

    stub("extensions.chamfer_distance.chamfer_distance", ChamferDistance=TrainingOnly)
    stub("extensions.earth_movers_distance.emd", EarthMoverDistance=TrainingOnly)
    stub("lib.vis")  # mayavi / skimage plotting, only called in debug branches
    try:
        import torchvision.transforms  # noqa: F401
    except ImportError:
        stub("torchvision")
        stub("torchvision.transforms")
    try:
        import scipy.misc  # noqa: F401
    except ImportError:
        stub("scipy.misc")

    # Their dataset hard-codes patients 1-21; read the list from a module variable instead
    path = os.path.join(repo, "datasets", "dataset.py")
    src = open(path).read()
    assert src.count(PATIENT_LINE) == 1, "dataset.py changed: patient list not found"
    mod = types.ModuleType("datasets.dataset")
    mod.__file__ = path
    mod.PATIENTS = None
    sys.modules["datasets.dataset"] = mod
    exec(compile(src.replace(PATIENT_LINE, "self.patients = PATIENTS"), path, "exec"), mod.__dict__)
    return mod


def load_config(repo, data_root, device):
    from easydict import EasyDict as edict
    from configs.models import architectures

    yaml.add_constructor("!join", lambda loader, node: "_".join(str(i) for i in loader.construct_sequence(node)),
                         Loader=yaml.Loader)
    with open(os.path.join(repo, "configs", "test", "main_config.yaml")) as f:
        config = edict(yaml.load(f, Loader=yaml.Loader))
    config.dataset = "syn"
    config.data_root = os.path.abspath(data_root)
    config.device = device
    config.kpfcn_config.architecture = architectures["syn"]
    return config


def build_model(config, weights):
    from models.lepard.lepard import Pipeline

    state = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    posenet = {k[len("posenet."):]: v for k, v in state.items() if k.startswith("posenet.")}
    model = Pipeline(config)
    model.load_state_dict(posenet, strict=True)
    return model.to(config.device).eval()


def to_device(batch, device):
    """Same rule as their tester loop."""
    for k, v in batch.items():
        if isinstance(v, list):
            if v and not isinstance(v[0], (str, np.ndarray)):
                batch[k] = [item.to(device) for item in v]
        elif not isinstance(v, (dict, float, type(None), np.ndarray)):
            batch[k] = v.to(device)
    return batch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="dataset/liver-registration")
    ap.add_argument("--patients", nargs="+", default=["01", "06", "10", "11"])
    ap.add_argument("--condition", choices=["A", "B"], required=True)
    ap.add_argument("--factor", type=float, default=None, help="fixed scale for B (default: recipe)")
    ap.add_argument("--limit", type=int, default=None, help="first N test views only (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ct-rot-z", type=float, default=0.0,
                    help="rotate the CT about its z axis by this many degrees before registration (frame test)")
    ap.add_argument("--ct-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="then shift the CT by this vector, metres (frame test)")
    ap.add_argument("--out", default="benchmark/selfp2ir_out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.Tensor.cuda = lambda self, *a, **k: self  # their solver calls .cuda() directly

    ds_mod = import_repo(os.path.abspath(args.repo))
    from datasets.dataloader import calibrate_neighbors, collate_fn_3dmatch
    from lib.logger import setup_seed
    from lib.loss import MatchMotionLoss as MML
    from models.lepard.matching import Matching as CM

    config = load_config(args.repo, args.data_root, device)
    patients = sorted(int(p) for p in args.patients)
    o3d.utility.random.seed(args.seed)

    tag = "A" if args.condition == "A" else ("B_recipe" if args.factor is None else f"B_x{args.factor:.2f}")
    a = np.deg2rad(args.ct_rot_z)
    G = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
    d = np.array(args.ct_shift, np.float32)
    if args.ct_rot_z or d.any():
        tag += f"_ctrot{args.ct_rot_z:g}_ctshift{'_'.join(f'{v:g}' for v in d)}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"predictions_{tag}.json")
    records, rejected, limits = [], [], None
    if os.path.exists(out_path):
        saved = json.load(open(out_path))
        if saved["meta"].get("complete"):
            print(f"{out_path} is complete; delete it to rerun")
            return
        records, rejected = saved["predictions"], [tuple(x) for x in saved["meta"]["rejected"]]
        limits = np.array(saved["meta"]["neighbourhood_limits"])
        print(f"resuming {out_path}: {len(records)} done, {len(rejected)} rejected")
    done = {(r["patient"], r["view"]) for r in records} | set(rejected)

    # Neighbourhood limits, as in main.py: augmented training set, patient 01's views first
    if limits is None:
        setup_seed(0)
        ds_mod.PATIENTS = [patients[0]]
        train_set = ds_mod.PoseDataset("train", config, data_augmentation=True)
        limits = calibrate_neighbors(train_set, config.kpfcn_config, collate_fn=collate_fn_3dmatch)
    print("neighbourhood limits:", limits.tolist())

    model = build_model(config, args.weights)
    ds_mod.PATIENTS = patients
    test_set = ds_mod.PoseDataset("test", config, data_augmentation=False)
    ct_vertices = {p: np.asarray(o3d.io.read_triangle_mesh(
        os.path.join(config.data_root, f"{p:02d}", "model", "reconstructed_mesh_world_m.obj")).vertices)
        for p in patients}

    meta = dict(
        condition=tag, scale_definition="range" if args.factor is None else "fixed",
        repo=os.path.abspath(args.repo), weights_sha256=sha256(args.weights),
        device=str(device), torch=torch.__version__, neighbourhood_limits=limits.tolist(),
        patients=patients, frames="R, t map CT points (metres) to the Blender camera (metres, -Z forward); "
                                  "R_true, t_true are the same from camPose.yml")

    def save(complete):
        with open(out_path, "w") as f:
            json.dump(dict(meta={**meta, "complete": complete, "n_views": len(records), "rejected": rejected},
                           predictions=records), f)

    n = len(test_set) if args.limit is None else min(args.limit, len(test_set))
    t0 = time.time()
    for i in range(n):
        patient, view = int(test_set.list_patient[i]), int(test_set.list_rank[i])
        if (patient, view) in done:
            continue
        np.random.seed(args.seed + i)  # their item sampling uses the global numpy RNG
        item = test_set.get_item(i)
        if item is None:
            rejected.append((patient, view))
            continue

        if args.ct_rot_z or d.any():  # move the CT; the true pose follows so that it still maps CT -> camera
            item["preope_pcd_src"] = item["preope_pcd_src"] @ G.T + d
            item["pose_R"] = item["pose_R"] @ G.T
            item["pose_t"] = item["pose_t"] - item["pose_R"] @ d.reshape(3, 1)
        scale = 1.0
        if args.condition == "B":
            if args.factor is None:
                raw = np.asarray(o3d.io.read_point_cloud(test_set.list_liverPcd[i]).points) / 1000.0
                scale = recipe_factor(ct_vertices[patient], raw)
            else:
                scale = args.factor
            item["intra_liver_pcd_tgt"] = item["intra_liver_pcd_tgt"] * np.float32(scale)

        batch = to_device(collate_fn_3dmatch([item], config.kpfcn_config, neighborhood_limits=limits), device)
        with torch.no_grad():
            data = model(batch)
            match_pred, _, _ = CM.get_match(data["conf_matrix_pred"], thr=0.15, mutual=False)
            rot, trn, _, _ = MML.ransac_regist_coarse(data["s_pcd"], data["t_pcd"], data["src_mask"],
                                                     data["tgt_mask"], match_pred, data)

        records.append(dict(
            patient=patient, view=view, scale=scale, n_matches=int(len(match_pred)),
            R=np.asarray(rot[0].detach().cpu(), float).tolist(),
            t=np.asarray(trn[0].detach().cpu(), float).reshape(3).tolist(),
            R_true=np.asarray(item["pose_R"], float).tolist(),
            t_true=np.asarray(item["pose_t"], float).reshape(3).tolist()))
        if len(records) % 25 == 0:
            save(complete=False)
            print(f"[{tag}] {i + 1}/{n} views, {len(rejected)} rejected, {time.time() - t0:.0f} s")

    save(complete=True)
    print(f"saved {len(records)} predictions ({len(rejected)} rejected) to {out_path}")


if __name__ == "__main__":
    main()
