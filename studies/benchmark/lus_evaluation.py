"""
Score a registration method on the LLR-LUS dataset: a Python port of the authors' Evaluation.m.

Rabbani et al., "A Methodology and Clinical Dataset with Ground-truth to Evaluate Registration
Accuracy Quantitatively in Computer-assisted Laparoscopic Liver Resection" (2021). The original
Matlab code is GPLv3; this port follows it step by step:
  - each LUS tumour profile (json, ultrasound pixels) is mapped to the camera frame (mm) with the
    image's calibration and probe pose (.mat), then resampled to 50 points along its outline;
  - TRE (per patient): 10 rounds of ICP between all the patient's profiles and the submitted tumours
    (one rotation and shift per patient, start: mean centroid offset) choose the closest tumour
    vertex for each profile point; TRE is the mean distance at the ORIGINAL positions;
  - IC (per patient): every profile point lies inside its tumour pushed out 10 mm from its centroid.

A method submits one tumour per image, <method-dir>/PatientN/<image>.stl, in OpenGL camera axes
(y and z are flipped here, as in Evaluation.m).

Usage: python benchmark/lus_evaluation.py --method-dir "dataset/lus-evaluation/results/Registration Methods/Koo"
"""
import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
import scipy.io as sio

DATASET = "dataset/lus-evaluation/data/Dataset"
K_ICP, M_SAMPLES, MARGIN_MM = 10, 50, 10.0


def read_stl(path):
    """Binary STL as unique vertices and faces (stlRead + stlSlimVerts)."""
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.int32)[0])
        rec = np.frombuffer(f.read(50 * n), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    V, F = np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0, return_inverse=True)
    return V, F.reshape(-1, 3)


def profile(dataset, p, name):
    """LUS tumour profile in the camera frame (mm), as 3 x n."""
    m = sio.loadmat(f"{dataset}/Patient{p}/LUS Calibration and Pose/{name}.mat")
    data = json.load(open(f"{dataset}/Patient{p}/LUS Segmentation/json/{name}.json"))
    uv = np.array(data["objects"][0]["points"]["exterior"], float).T
    us = np.vstack([m["SLus"][0, 0] * uv[0], np.zeros(uv.shape[1]), m["SAus"][0, 0] * uv[1]])
    return m["Rpr"] @ (m["Rus"] @ us + m["Tus"]) + m["Tpr"]


def sample_profile(P, M):
    """M points at equal steps along the open polyline P (3 x n), starting at its first point."""
    seg = np.linalg.norm(np.diff(P, axis=1), axis=0)
    s = np.r_[0, np.cumsum(seg)]
    at = np.arange(M) * s[-1] / M
    return np.stack([np.interp(at, s, P[k]) for k in range(3)])


def absor(A, B):
    """Rigid R, t minimising sum ||R A + t - B||^2 (Horn; A, B are 3 x n)."""
    ma, mb = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd((B - mb) @ (A - ma).T)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, mb - R @ ma


def tre(P, tumours):
    """P: per image 3 x M profile points; tumours: per image vertices. Mean distance to the ICP-chosen points."""
    nearest = lambda Q, V: V[((Q.T[:, None] - V[None]) ** 2).sum(2).argmin(1)].T  # noqa: E731 (knnsearch)
    R = np.eye(3)
    t = np.mean([V.mean(0)[:, None] - Pi.mean(1, keepdims=True) for Pi, V in zip(P, tumours)], axis=0)
    for _ in range(K_ICP):
        T = [nearest(R @ Pi + t, V) for Pi, V in zip(P, tumours)]
        R, t = absor(np.hstack(P), np.hstack(T))
    return np.linalg.norm(np.hstack(P) - np.hstack(T), axis=0).mean()


def inclusion(P, tumours, faces):
    """Every profile point inside its tumour pushed out MARGIN_MM from the tumour's vertex centroid."""
    for Pi, V, F in zip(P, tumours, faces):
        c = V.mean(0)
        grown = V + MARGIN_MM * (V - c) / np.linalg.norm(V - c, axis=1, keepdims=True)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(grown.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        if not scene.compute_occupancy(o3d.core.Tensor(Pi.T.astype(np.float32))).numpy().all():
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True, help="folder with PatientN/<image>.stl")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    tres, ics = [], []
    for p in range(1, 5):
        names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{args.dataset}/Patient{p}/LUS Calibration and Pose/*.mat"))
        P, tumours, faces = [], [], []
        for n in names:
            V, F = read_stl(f"{args.method_dir}/Patient{p}/{n}.stl")
            V[:, 1:] *= -1  # OpenGL camera axes -> OpenCV camera axes
            tumours.append(V)
            faces.append(F)
            P.append(sample_profile(profile(args.dataset, p, n), M_SAMPLES))
        tres.append(tre(P, tumours))
        ics.append(inclusion(P, tumours, faces))
        print(f"Patient {p} ({len(names)} images): TRE = {tres[-1]:.2f} mm, IC = {'Pass' if ics[-1] else 'Failed'}")
    print(f"Overall: TRE = {np.mean(tres):.2f} mm, IC = {sum(ics)} of 4")


if __name__ == "__main__":
    main()
