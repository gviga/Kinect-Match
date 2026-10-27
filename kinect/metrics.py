"""Errors of a point-to-point map between two clean clouds.

A map from X to Y is an integer array ``T`` with one entry per clean point of
X: ``T[i]`` is the index, in Y's clean cloud, of the image of point ``i``.

Geodesic distances on a point cloud are taken on its SMPL registration: each
point stands for its nearest SMPL vertex, and errors are divided by the
geodesic diameter of the target registration.
"""

import numpy as np
import potpourri3d as pp3d


class Geodesics:
    """All-pairs geodesic distances on one SMPL registration (heat method)."""

    def __init__(self, shape):
        solver = pp3d.MeshHeatMethodDistanceSolver(shape["smpl_v"].astype(np.float64), shape["smpl_f"])
        n = len(shape["smpl_v"])
        self.D = np.stack([solver.compute_distance(i) for i in range(n)]).astype(np.float32)
        self.D = 0.5 * (self.D + self.D.T)  # the heat method is not exactly symmetric
        self.diam = float(self.D.max())
        self.vertex = shape["pc2smpl"][shape["inlier"]]  # clean point -> SMPL vertex


def _errors(T, src, gt, shape_y, geo):
    clean_y = shape_y["points"][shape_y["inlier"]]
    pred = T[src]
    euc = np.linalg.norm(clean_y[pred] - clean_y[gt], axis=1)
    out = dict(euc=float(euc.mean()), n=int(len(src)))
    if geo is not None:
        out["geo"] = float((geo.D[geo.vertex[pred], geo.vertex[gt]] / geo.diam).mean())
    return out


def evaluate(T, shape_x, shape_y, pair=None, geo=None):
    """Errors of ``T`` at every ground-truth level.

    smpl         the NAM / FUSE protocol: the SMPL vertices, through the .vts files
    smpl_visible the same, on vertices seen in both clouds
    dense        every clean point of X (needs ``pair`` from ``kinect.pairs.pair_gt``)
    dense_valid  the same, where the ground truth is valid (partiality)
    """
    T = np.asarray(T).reshape(-1)
    n_x, n_y = int(shape_x["inlier"].sum()), int(shape_y["inlier"].sum())
    if len(T) != n_x:
        raise ValueError(f"map has {len(T)} entries, the clean source cloud has {n_x} points")
    res = dict(coverage=float(len(np.unique(T)) / n_y))
    sx, sy = shape_x["smpl2clean"], shape_y["smpl2clean"]
    vis = shape_x["smpl_visible"] & shape_y["smpl_visible"]
    res["smpl"] = _errors(T, sx, sy, shape_y, geo)
    res["smpl_visible"] = _errors(T, sx[vis], sy[vis], shape_y, geo)
    if pair is not None:
        src = np.arange(n_x)
        res["dense"] = _errors(T, src, pair["gt"], shape_y, geo)
        v = pair["valid"]
        res["dense_valid"] = _errors(T, src[v], pair["gt"][v], shape_y, geo)
    return res
