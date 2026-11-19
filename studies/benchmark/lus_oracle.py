"""
Oracle scale for the LLR-LUS clouds: the sigma that puts each cloud at the TRUE distance from the camera.

Self-P2IR's pipeline sets the cloud's distance with its scale step (sigma = CT length / cloud length), which
ignores the camera. The oracle replaces only that number, using ground truth: the ultrasound probe's face
touches the liver, and the tracked probe pose puts that face in the camera frame (same calibration as the
ground-truth tumour). Moving a cloud along its viewing rays (one factor on every point's distance from the
camera) changes its distance and size together and leaves it unchanged in the image.

Per image:
  contact depth   the probe face (an arc at a fixed place in the scanner screenshot, the same for all four
                  patients) mapped to the camera frame: median depth, mm
  cloud depth     the cloud's depth (Self-P2IR's recipe) on the visible liver pixels within NEAR of the
                  face's projection (the probe hides the liver at the contact itself)
  oracle sigma    sigma * contact depth / cloud depth
This uses the probe tracking, which also produces the ground truth, so it is an UPPER BOUND on what a correct
distance could give, never a method.

Writes dataset/lus-evaluation/selfp2ir/oracle.json (also each image's "near" region as run lengths, so a depth model's
depth can be compared with the contact depth on the same pixels: depth_benchmark.py).
Usage: python studies/benchmark/lus_oracle.py
"""
import json

import cv2
import numpy as np
import open3d as o3d
import scipy.io as sio

LUS = "dataset/lus-evaluation"
D = f"{LUS}/data/Dataset"
NEAR = 0.05  # liver pixels within this fraction of the image width from the projected probe face


def probe_face():
    """The probe's contact face in the scanner screenshot (pixels): the topmost bright pixel per column of the arc."""
    us = cv2.imread(f"{D}/Patient1/LUS Images/02.jpg", 0).astype(int)
    arc = [(c, 85 + int(np.argmax(us[85:125, c] > 120))) for c in range(340, 536) if (us[85:125, c] > 120).any()]
    return np.array(arc, float)


def main():
    meta = json.load(open(f"{LUS}/selfp2ir/meta.json"))
    arc = probe_face()
    out = {}
    for key, m in sorted(meta.items()):
        p, name = key.split("/")
        cal = sio.loadmat(f"{D}/{p}/LUS Calibration and Pose/{name}.mat")
        U = np.vstack([cal["SLus"][0, 0] * arc[:, 0], np.zeros(len(arc)), cal["SAus"][0, 0] * arc[:, 1]])
        face = (cal["Rpr"] @ (cal["Rus"] @ U + cal["Tus"]) + cal["Tpr"]).T                      # camera frame, mm
        u = m["fx"] * face[:, 0] / face[:, 2] + m["cx"]
        v = m["fy"] * face[:, 1] / face[:, 2] + m["cy"]

        W, H = m["size"]
        liver = cv2.imread(f"{LUS}/masks_manual/{p}/{name}.png", 0) > 0
        line = np.zeros((H, W), np.uint8)
        cv2.polylines(line, [np.stack([u, v], 1).round().astype(np.int32)], False, 1, 1)
        dist = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
        near = liver & (dist <= NEAR * W)
        cloud = np.asarray(o3d.io.read_point_cloud(f"{LUS}/selfp2ir/{m['id']}/real/liverPcds/{m['frame']}.ply").points)
        z = (cloud[:, 2] + m["shift_mm"][2]).reshape(H, W)                                         # camera-frame depth
        contact, cloud_depth = float(np.median(face[:, 2])), float(np.median(z[near]))
        flat = near.reshape(-1).astype(np.int8)                                              # run lengths, row order,
        change = np.flatnonzero(np.diff(flat)) + 1                                           # starting with background
        runs = np.diff(np.r_[0, change, flat.size]).tolist()
        out[key] = dict(sigma=m["sigma"] * contact / cloud_depth, sigma_recipe=m["sigma"], contact_depth_mm=contact,
                        cloud_depth_near_mm=cloud_depth, near_pixels=int(near.sum()),
                        near_rle=([0] if flat[0] else []) + runs, size=[W, H])
        print(f"{key}: contact {contact:5.0f} mm, cloud near the probe {cloud_depth:5.0f} mm -> sigma x {contact / cloud_depth:.2f} "
              f"({int(near.sum())} liver pixels near the probe)", flush=True)
    json.dump(out, open(f"{LUS}/selfp2ir/oracle.json", "w"), indent=1)
    f = np.array([o["sigma"] / o["sigma_recipe"] for o in out.values()])
    print(f"wrote {LUS}/selfp2ir/oracle.json: the oracle rescales clouds by x{np.median(f):.2f} (median; range {f.min():.2f}-{f.max():.2f})")


if __name__ == "__main__":
    main()
