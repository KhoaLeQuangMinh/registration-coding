"""
3D ridge of a CT liver, found from its shape alone (no cameras, no scanner axes).

The liver has two surfaces: the smooth dome under the diaphragm and the grooved underside that faces
the gut. They meet at a rim that is thin and sharp at the front (the inferior border) and rounded at
the back. The ridge is the sharp part of that rim:
  1. fold   how sharply the surface bends within 8 mm of each vertex (+ convex edge, - groove)
  2. split  a minimum cut between dome seeds (outer shell, facing away from the underside) and
            underside seeds (the deepest pits, and surface facing the same way as them); cutting is
            cheap where the surface folds sharply, so the cut follows the rim
  3. ridge  rim vertices that fold more than 95 % of the dome does, extended along the rim while they
            fold more than 80 % of it; pieces shorter than 3 cm are dropped

Frozen on 2026-11-10, before the one-time check against Koo et al.'s expert ridge on the LLR-LUS
livers. Do not tune it on LLR-LUS.

Saved to registration/landmarks/<patient>.json as {"ridge": [[vertex indices of one piece, in order], ...]}
on the CT mesh as loaded by benchmark/pilot_dice_vs_tumour.py (mm).

Usage: python registration/ct_landmarks.py [--patients 01 06 10 11] [--show]
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra, maximum_flow

from capture_range import HERE
from pilot_dice_vs_tumour import DATA_DIR

FOLD_MM = 8.0
START, KEEP = 95, 80   # percentiles of the dome's fold
MIN_PIECE_MM = 30.0


def load_mesh(pid: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(os.path.join(DATA_DIR, pid, "model", "reconstructed_mesh_world_m.obj"))
    mesh.scale(1000.0, center=[0, 0, 0])
    mesh.compute_vertex_normals()
    return mesh


def fold(V, N):
    """Signed fold per vertex: spread of the normals within FOLD_MM (0 flat .. 1), + convex, - concave."""
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V)))
    f = np.zeros(len(V))
    for i in range(len(V)):
        nb = np.asarray(tree.search_radius_vector_3d(V[i], FOLD_MM)[1])
        f[i] = (1 - np.linalg.norm(N[nb].mean(0))) * np.sign(np.sum((V[nb] - V[i]) * (N[nb] - N[i])))
    return f


def hull_depth(mesh):
    """How far each vertex lies inside the convex hull (mm): ~0 on the outer shell, large in pits."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.compute_convex_hull()[0]))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32))).numpy()


def graph(V, F):
    """Symmetric vertex graph of the mesh, weighted by edge length (mm)."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    G = sp.coo_matrix((np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1), (e[:, 0], e[:, 1])), shape=(len(V),) * 2).tocsr()
    return G.maximum(G.T)


def split(V, N, F, G, f, depth):
    """True for dome vertices: minimum cut between dome and underside seeds, cheap across sharp convex folds."""
    i, j = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0).T  # every edge once
    sharp = np.clip(f[[i, j]].mean(0), 0, None) / np.percentile(np.abs(f), 98)
    cap = np.maximum(1, np.round(100 * np.linalg.norm(V[i] - V[j], axis=1) * np.exp(-5 * sharp))).astype(np.int32)
    pits = depth > np.percentile(depth, 90)   # porta hepatis, gallbladder bed, fissures: always underside
    u = V[pits].mean(0) - V.mean(0)
    u /= np.linalg.norm(u)                    # the way the underside faces
    dome_seed = np.flatnonzero((depth < 1.0) & (N @ u < -0.5))
    under_seed = np.flatnonzero(pits | (N @ u > 0.5))
    n, big = len(V), np.int32(1 << 30)
    rows = np.r_[i, j, np.full(len(dome_seed), n), under_seed]
    cols = np.r_[j, i, dome_seed, np.full(len(under_seed), n + 1)]
    vals = np.r_[cap, cap, np.full(len(dome_seed) + len(under_seed), big)].astype(np.int32)
    C = sp.csr_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))
    res = C - maximum_flow(C, n, n + 1).flow
    res.data[res.data < 0] = 0
    res.eliminate_zeros()
    dome = np.zeros(n + 2, bool)
    dome[breadth_first_order(res, n, return_predecessors=False)] = True
    dome = dome[:n]
    for side in (True, False):  # each surface in one piece: small islands (e.g. a pit in the dome) join the other side
        m = dome == side
        _, lab = connected_components(G[m][:, m], directed=False)
        dome[np.flatnonzero(m)[lab != np.bincount(lab).argmax()]] = not side
    return dome


def longest_path(G, nodes):
    """Vertices of the longest shortest path through `nodes` (assumed connected), in order."""
    Gb = G[nodes][:, nodes]
    d = dijkstra(Gb, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d), d, -1)))
    d, pred = dijkstra(Gb, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(d), d, -1)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return nodes[path]


def ridge(mesh) -> list:
    """Ridge pieces of a liver mesh in mm (with vertex normals), each an ordered array of vertex indices."""
    V, N, F = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), np.asarray(mesh.triangles)
    G = graph(V, F)
    f = fold(V, N)
    dome = split(V, N, F, G, f, hull_depth(mesh))
    i, j = G.nonzero()
    rim = np.unique(i[dome[i] & ~dome[j]])   # dome side of the cut
    weak = rim[f[rim] > np.percentile(f[dome], KEEP)]
    strong = rim[f[rim] > np.percentile(f[dome], START)]
    _, lab = connected_components(G[weak][:, weak], directed=False)
    sharp = weak[np.isin(lab, lab[np.isin(weak, strong)])]
    _, lab = connected_components(G[sharp][:, sharp], directed=False)
    pieces = [longest_path(G, sharp[lab == c]) for c in range(lab.max() + 1)]
    return [p for p in pieces if np.linalg.norm(np.diff(V[p], axis=0), axis=1).sum() >= MIN_PIECE_MM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["01", "06", "10", "11"])
    ap.add_argument("--show", action="store_true", help="open a 3D window per patient (grey liver, red ridge)")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "landmarks"), exist_ok=True)
    for pid in args.patients:
        mesh = load_mesh(pid)
        pieces = ridge(mesh)
        V = np.asarray(mesh.vertices)
        json.dump({"ridge": [p.tolist() for p in pieces]}, open(os.path.join(HERE, "landmarks", f"{pid}.json"), "w"))
        print(f"{pid}: ridge pieces (mm) " + ", ".join(f"{np.linalg.norm(np.diff(V[p], axis=0), axis=1).sum():.0f}" for p in pieces))
        if args.show:
            colors = np.full((len(V), 3), 0.75)
            tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(mesh.vertices))
            for i in np.concatenate(pieces):
                colors[tree.search_radius_vector_3d(V[i], 2.5)[1]] = [0.9, 0.1, 0.1]
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([mesh], window_name=f"patient {pid}: ridge (red)")


if __name__ == "__main__":
    main()
