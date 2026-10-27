"""Level (c): dense point-to-point ground truth between two shapes.

Every clean point of X is carried to Y through the SMPL surface: same face,
same barycentric coordinates, same offset along the (now deformed) face
normal. Its ground-truth image is the nearest clean point of Y to that
location. Where Y did not see that part of the body the nearest point is far
away, and the correspondence is marked invalid.
"""

import numpy as np
import trimesh
from scipy.spatial import KDTree

from .gt import TAU


def transfer(shape_x, shape_y):
    """Locations on Y of the clean points of X, (M_x, 3)."""
    m = shape_x["inlier"]
    face, bary, offset = shape_x["face"][m], shape_x["bary"][m], shape_x["offset"][m]
    mesh = trimesh.Trimesh(shape_y["smpl_v"].astype(np.float64), shape_y["smpl_f"], process=False)
    tri = mesh.triangles[face]
    return np.einsum("ij,ijk->ik", bary, tri) + offset[:, None] * mesh.face_normals[face]


def pair_gt(shape_x, shape_y, tau=TAU):
    """Ground truth from the clean cloud of X to the clean cloud of Y.

    Returns gt (M_x,) indices into Y's clean cloud, valid (M_x,) bool and the
    distance (M_x,) from the transferred location to the chosen point."""
    q = transfer(shape_x, shape_y)
    clean_y = shape_y["points"][shape_y["inlier"]].astype(np.float64)
    dist, gt = KDTree(clean_y).query(q)
    return dict(gt=gt.astype(np.int32), valid=dist < tau, dist=dist.astype(np.float32))
