"""Ground truth between a Kinect point cloud and its SMPL registration.

Levels
------
(a) raw      SMPL vertex <-> point nearest-neighbour maps on the raw 10k cloud.
(b) clean    the same after removing points far from the SMPL surface; this is
             the ``off_clean`` / ``corres`` test set used in NAM and FUSE.
(c) dense    every point expressed on the SMPL surface (face, barycentric
             coordinates, signed offset along the face normal), from which
             point-to-point ground truth between any two shapes follows
             (``kinect.pairs``).
masks        ``inlier`` per point; ``smpl_visible`` per SMPL vertex, true where
             the clean cloud has a point within ``tau`` metres, i.e. where the
             ground truth is reliable on a partial acquisition.
"""

import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.stats import zscore

N_POINTS = 10_000
Z_THRESHOLD = 1.0  # value used for the NAM / FUSE test set
TAU = 0.03  # metres; SMPL vertices farther than this from every clean point are not visible


def subsample(n_scan, n=N_POINTS, seed=0):
    """Indices of ``n`` points drawn uniformly, with replacement, from a scan of
    ``n_scan`` points (the test set was drawn the same way, without a seed)."""
    return np.random.default_rng(seed).integers(0, n_scan, n)


def nearest(src, query):
    """Index into ``src`` of the nearest neighbour of every row of ``query``."""
    return KDTree(src).query(query)[1]


def inlier_mask(points, smpl_v, z=Z_THRESHOLD, two_sided=True):
    """Points whose distance to the nearest SMPL vertex has |z-score| < z.

    ``two_sided=True`` reproduces the test set; it also drops points unusually
    close to the surface. ``two_sided=False`` keeps those."""
    s = zscore(KDTree(smpl_v).query(points)[0])
    return np.abs(s) < z if two_sided else s < z


def surface_coords(points, smpl_v, smpl_f):
    """Closest point on the SMPL surface of every point, as (face index,
    barycentric coordinates, signed offset along the face normal)."""
    mesh = trimesh.Trimesh(smpl_v.astype(np.float64), smpl_f, process=False)
    closest, _, face = trimesh.proximity.closest_point(mesh, points.astype(np.float64))
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face], closest)
    offset = np.einsum("ij,ij->i", points - closest, mesh.face_normals[face])
    return face.astype(np.int32), bary.astype(np.float32), offset.astype(np.float32)


def build_shape(scan, smpl_v, smpl_f, idx, z=Z_THRESHOLD, two_sided=True, tau=TAU, dense=True):
    """All ground-truth levels of one shape.

    scan  (P, 3)  BEHAVE ``person.ply`` points
    idx   (N,)    indices of the subsample into ``scan``
    """
    points = scan[idx].astype(np.float32)  # pc_i.ply stored float32
    p, v = points.astype(np.float64), smpl_v.astype(np.float64)

    smpl2pc = nearest(p, v)  # (a) p2p_tri
    pc2smpl = nearest(v, p)  # (a) p2p_pc
    inlier = inlier_mask(p, v, z, two_sided)
    clean = p[inlier]
    smpl2clean = nearest(clean, p[smpl2pc])  # (b) the released .vts
    dist = KDTree(clean).query(v)[0]

    out = dict(
        points=points, scan_idx=idx.astype(np.int32),
        smpl_v=smpl_v.astype(np.float32), smpl_f=smpl_f.astype(np.int32),
        smpl2pc=smpl2pc.astype(np.int32), pc2smpl=pc2smpl.astype(np.int32),
        inlier=inlier, smpl2clean=smpl2clean.astype(np.int32),
        smpl_dist=dist.astype(np.float32), smpl_visible=dist < tau,
        params=np.array([z, float(two_sided), tau], dtype=np.float32),
    )
    if dense:
        out["face"], out["bary"], out["offset"] = surface_coords(p, v, smpl_f)
    return out
