"""
Mark 3D landmarks on a CT liver by hand: the ridge and the falciform ligament groove.

Marking window: shift + left click picks a point, shift + right click undoes the last one,
Q closes and saves. Click along the whole landmark, in order, about every 5-10 mm:
  ridge     the sharp lower front edge of the liver, across both lobes, from end to end
  ligament  the groove where the falciform ligament joins the liver, from the top down to the
            notch in the lower edge
Landmarks already marked are shown in colour. Marking a landmark again replaces it.

Saved to registration/landmarks/<patient>.json as vertex indices of the CT mesh (in mm, as loaded
by benchmark/pilot_dice_vs_tumour.py). A review window then shows the marked points on the mesh.

Usage: python registration/mark_landmarks.py --patient 06 --landmark ridge
       python registration/mark_landmarks.py --patient 06            (review only)
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

from capture_range import HERE
from pilot_dice_vs_tumour import DATA_DIR
from show_module3 import BLUE, GREY, RED, ball

COLORS = {"ridge": RED, "ligament": BLUE}


def load_mesh(pid: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(os.path.join(DATA_DIR, pid, "model", "reconstructed_mesh_world_m.obj"))
    mesh.scale(1000.0, center=[0, 0, 0])
    mesh.compute_vertex_normals()
    return mesh


def pick(mesh, marks: dict, landmark: str) -> list:
    """Vertex indices clicked in the window, in click order."""
    V = np.asarray(mesh.vertices)
    cloud = o3d.geometry.PointCloud(mesh.vertices)
    cloud.normals = mesh.vertex_normals
    colors = np.tile(GREY, (len(V), 1))
    tree = o3d.geometry.KDTreeFlann(cloud)
    for name, idx in marks.items():
        for i in idx:  # colour a 2 mm patch around each marked vertex so it is visible
            colors[tree.search_radius_vector_3d(V[i], 2.0)[1]] = COLORS[name]
    cloud.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"mark the {landmark}: shift+click points in order, shift+right click undoes, Q saves")
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="06")
    ap.add_argument("--landmark", choices=list(COLORS), help="landmark to mark; omit to only review")
    args = ap.parse_args()

    path = os.path.join(HERE, "landmarks", f"{args.patient}.json")
    marks = json.load(open(path)) if os.path.exists(path) else {}
    mesh = load_mesh(args.patient)
    if args.landmark:
        picked = pick(mesh, {k: v for k, v in marks.items() if k != args.landmark}, args.landmark)
        if len(picked) < 2:
            print(f"only {len(picked)} point(s) picked; nothing saved")
            return
        marks[args.landmark] = picked
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(marks, open(path, "w"), indent=1)
        print(f"saved {len(picked)} {args.landmark} points to {path}")

    V = np.asarray(mesh.vertices)
    geoms = [mesh.paint_uniform_color(GREY)]
    for name, idx in marks.items():
        geoms.append(o3d.geometry.LineSet(o3d.utility.Vector3dVector(V[idx]),
                                          o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(idx) - 1)])))
        geoms[-1].paint_uniform_color(COLORS[name])
        geoms += [ball(V[i], COLORS[name], 1.5) for i in idx]
        print(f"{name}: {len(idx)} points, curve length {np.linalg.norm(np.diff(V[idx], axis=0), axis=1).sum():.0f} mm")
    o3d.visualization.draw_geometries(geoms, window_name=f"patient {args.patient}: ridge (red), ligament (blue)")


if __name__ == "__main__":
    main()
