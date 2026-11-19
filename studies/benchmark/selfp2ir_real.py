"""
Run the authors' pretrained Self-P2IR on P2I-LReg real test frames; save their Dice and Chamfer per frame.

Purpose: check that the released weights reproduce the paper before the model is used elsewhere.
Paper (Zhou et al., TMI 2025, Table II, mean of five folds): Dice 78.89 +- 6.76 %, CD 2.97 +- 1.06.
The released weights are one fold, so the result should fall near that range, not on it.

Follows their real tester (tester.py, _realTester.test_thr) step by step:
  dataset item -> collate_fn_3dmatch -> DeformRegis: rigid network, then the deformation pyramid
  optimised on this frame, then the bent CT points drawn into the image -> Dice with the liver label,
  CD between the frame's cloud and the bent CT points.
Neighbourhood limits are calibrated as in main.py, on the real training set with setup_seed(0).

Deviations from their code, all needed to run it on a current PyTorch:
  - Their PointNet++ CUDA extension is replaced by tools/pointnet2_torch.py (same operations in
    plain PyTorch) and their Chamfer CUDA extension by `ChamferDistance` below (same numbers:
    squared nearest distances in both directions).
  - Plotting and training-only modules are replaced by stubs, as in selfp2ir_predict.py.
  - Weights are loaded with strict=True for every tensor the file holds (their loader: strict=False).
  - A frame their loader rejects (< 100 liver pixels) is skipped and recorded.
  - The rendered depth is not turned into points: that output is used in training only, and their
    helper (lib/geometry.depth2pointcloud) is GPU-only and fails when the drawn liver is small.
  - Without a GPU, tensor.cuda() is a no-op.

Besides their two numbers, two plain distances are saved per frame, in mm (not squared):
  cloud_to_ct_mm  mean distance from the frame's cloud to the nearest bent CT point
  ct_to_cloud_mm  mean distance from every CT point (the whole liver) to the nearest cloud point
Their "CD" is mean(squared metres, both directions added) x 1000.

Usage:
  python studies/benchmark/selfp2ir_real.py --repo <Self-P2IR checkout> --weights model_best_loss.pth \
      [--data-root dataset/liver-registration] [--patients 01 02 ...] [--limit N] [--out studies/benchmark/selfp2ir_real_out]
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfp2ir_predict import import_repo, load_config, sha256, stub, to_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class ChamferDistance(torch.nn.Module):
    """Squared distance from each point to its nearest point in the other set, both ways (as chamfer3D.cu)."""

    def forward(self, a, b):
        d = torch.cdist(a, b) ** 2
        return d.min(dim=2).values, d.min(dim=1).values


def optional(name, **attrs):
    try:
        importlib.import_module(name)
    except ImportError:
        stub(name, **attrs)


def import_real(repo, device):
    """Their real-frame modules, importable on a current PyTorch."""
    ds_mod = import_repo(repo)
    stub("extensions.chamfer_distance.chamfer_distance", ChamferDistance=ChamferDistance)
    stub("lib.vis", project_mesh_to_image=None, visualize_pcds=None)
    for name, attrs in [("skimage", dict(io=None, img_as_ubyte=None)), ("imageio", {}), ("h5py", {}),
                        ("geomloss", dict(SamplesLoss=None)), ("tqdm.notebook", dict(tqdm=None))]:
        optional(name, **attrs)
    spec = importlib.util.spec_from_file_location("models.pointnet.pointnet2.pointnet2_utils",
                                                  os.path.join(HERE, "tools", "pointnet2_torch.py"))
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    sys.modules[spec.name] = ops

    import models.mynetwork as net
    net.depth2pointcloud = lambda depth, K, downsampling, sampling_points: \
        torch.zeros(depth.shape[0], sampling_points, 3, device=depth.device)
    if device.type == "cpu":
        torch.Tensor.cuda = lambda self, *a, **k: self
    return ds_mod, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="dataset/liver-registration")
    ap.add_argument("--patients", nargs="+", default=None, help="default: every patient in the data root")
    ap.add_argument("--limit", type=int, default=None, help="first N test frames only (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-points", default=None,
                    help="folder: per frame an .npz with R, t, the bent CT points and the cloud points (their frame: "
                         "Blender camera axes, metres, after the cloud's shift) and the rendered mask, for viewing")
    ap.add_argument("--limits", type=int, nargs=4, default=None,
                    help="neighbourhood limits to use instead of calibrating them on the data root's training frames "
                         "(LLR-LUS has none: use the P2I-LReg calibration, [9, 19, 25, 30] on all 21 patients)")
    ap.add_argument("--out", default=os.path.join(HERE, "selfp2ir_real_out"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_mod, net = import_real(os.path.abspath(args.repo), device)
    from configs.models import architectures
    from datasets.dataloader import calibrate_neighbors, collate_fn_3dmatch
    from lib.logger import setup_seed
    from lib.loss import BinaryDiceLoss

    config = load_config(args.repo, args.data_root, device)
    config.dataset = "real"
    config.kpfcn_config.architecture = architectures["real"]
    patients = sorted(int(p) for p in (args.patients or
                                       [d for d in os.listdir(config.data_root) if d.isdigit()]))
    o3d.utility.random.seed(args.seed)

    setup_seed(0)
    ds_mod.PATIENTS = patients
    if args.limits:
        limits = np.array(args.limits)
    else:
        train_set = ds_mod.PoseDataset("train", config, data_augmentation=True)
        limits = calibrate_neighbors(train_set, config.kpfcn_config, collate_fn=collate_fn_3dmatch)
    print("neighbourhood limits:", limits.tolist())

    model = net.DeformRegis(config)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not missing, f"weights do not match the model: missing {missing}, unexpected {unexpected}"
    model = model.to(device).eval()
    for name, param in model.named_parameters():  # as their tester
        param.requires_grad = "deformNet" in name

    test_set = ds_mod.PoseDataset("test", config, data_augmentation=False)
    dice_loss, chamfer = BinaryDiceLoss(), ChamferDistance()
    records, rejected = [], []
    n = len(test_set) if args.limit is None else min(args.limit, len(test_set))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "real_frames.json")
    t0 = time.time()
    for i in range(n):
        patient = int(test_set.list_patient[i])
        frame = os.path.basename(os.path.dirname(test_set.list_label[i]))
        np.random.seed(args.seed + i)  # their item sampling uses the global numpy RNG
        item = test_set.get_item(i)
        if item is None:
            rejected.append((patient, frame))
            continue
        batch = to_device(collate_fn_3dmatch([item], config.kpfcn_config, neighborhood_limits=limits), device)
        data = model(batch)
        with torch.no_grad():
            R, t = data["R_s2t_pred"], data["t_s2t_pred"]
            bent = torch.bmm(R, data["deformed_src"].permute(0, 2, 1)).permute(0, 2, 1) + t.permute(0, 2, 1)
            d_cloud, d_ct = chamfer(data["batched_tgt_pcd"], bent)
            records.append(dict(
                patient=patient, frame=frame, scale=float(item["scale"]),
                dice=float(100.0 * (1.0 - dice_loss(data["mask"], data["liver_label"]))),
                cd=float(((d_cloud + d_ct) * 1000.0).mean()),
                cloud_to_ct_mm=float(d_cloud.sqrt().mean() * 1000.0),
                ct_to_cloud_mm=float(d_ct.sqrt().mean() * 1000.0),
                R=R[0].cpu().numpy().astype(float).tolist(), t=t[0].cpu().numpy().astype(float).reshape(3).tolist()))
            if args.save_points:
                os.makedirs(args.save_points, exist_ok=True)
                rendered = (data["mask"][0] > 0).cpu().numpy()
                np.savez_compressed(os.path.join(args.save_points, f"{patient:02d}_{frame}.npz"),
                                    R=R[0].cpu().numpy(), t=t[0].cpu().numpy(), bent=bent[0].cpu().numpy(),
                                    cloud=data["batched_tgt_pcd"][0].cpu().numpy(),
                                    rendered_mask=np.packbits(rendered), mask_shape=np.array(rendered.shape))
        r = records[-1]
        print(f"patient {patient:02d} {frame}: dice {r['dice']:.1f}  cd {r['cd']:.2f}  "
              f"cloud->ct {r['cloud_to_ct_mm']:.1f} mm  ct->cloud {r['ct_to_cloud_mm']:.1f} mm  "
              f"({time.time() - t0:.0f} s)", flush=True)
        with open(out_path, "w") as f:
            json.dump(dict(meta=dict(repo=os.path.abspath(args.repo), weights_sha256=sha256(args.weights),
                                     device=str(device), torch=torch.__version__, patients=patients,
                                     neighbourhood_limits=limits.tolist(), rejected=rejected,
                                     complete=i == n - 1), frames=records), f)

    dice, cd = np.array([r["dice"] for r in records]), np.array([r["cd"] for r in records])
    print(f"\n{len(records)} frames from patients {sorted({r['patient'] for r in records})}, {len(rejected)} rejected")
    print(f"Dice {dice.mean():.2f} +- {dice.std():.2f} %   (paper, five folds: 78.89 +- 6.76)")
    print(f"CD   {cd.mean():.2f} +- {cd.std():.2f}     (paper, five folds: 2.97 +- 1.06)")
    for key in ("cloud_to_ct_mm", "ct_to_cloud_mm"):
        print(f"{key}: mean {np.mean([r[key] for r in records]):.1f} mm")
    print("saved", out_path)


if __name__ == "__main__":
    main()
