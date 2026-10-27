"""Locating frames inside a local copy of BEHAVE.

A frame is named by its sequence and time stamp, e.g.
``Date01_Sub01_stool_sit/t0040.000``. Its files are

    <seq>/<frame>/person/person.ply              fused Kinect point cloud of the person
    <seq>/<frame>/person/<fit>/person_fit.ply    SMPL registration (6890 vertices)

The sequences may sit directly under the root (official ``sequences/`` folder)
or grouped by date (``Date01/<seq>``); both layouts are found.
"""

from pathlib import Path

import numpy as np
import trimesh


def frame_dir(root, frame):
    root = Path(root)
    seq, t = frame.split("/")
    for cand in (root / seq / t, root / "sequences" / seq / t, root / seq.split("_")[0] / seq / t):
        if (cand / "person").is_dir():
            return cand
    raise FileNotFoundError(f"BEHAVE frame {frame} not found under {root}")


def load_frame(root, frame, fit="fit02"):
    """(scan points (P, 3), SMPL vertices (6890, 3), SMPL faces (13776, 3)) of a frame,
    in the BEHAVE camera frame (metres)."""
    d = frame_dir(root, frame) / "person"
    scan = trimesh.load(d / "person.ply", process=False)
    reg = trimesh.load(d / fit / "person_fit.ply", process=False)
    return (np.asarray(scan.vertices, dtype=np.float64),
            np.asarray(reg.vertices, dtype=np.float32),
            np.asarray(reg.faces, dtype=np.int32))


def list_frames(root, fit="fit02"):
    """Every frame under ``root`` that has both a scan and a ``fit`` registration."""
    out = []
    for p in sorted(Path(root).rglob(f"person/{fit}/person_fit.ply")):
        frame_path = p.parents[2]
        if (frame_path / "person" / "person.ply").exists():
            out.append(f"{frame_path.parent.name}/{frame_path.name}")
    return out
