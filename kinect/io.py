"""Reading and writing the dataset files.

Formats
-------
- ``.off``  point cloud, no faces, one ``x y z`` row per point (as written by meshio).
- ``.vts``  one 0-based point index per SMPL vertex (6890 rows).
- ``.npz``  every ground-truth level of one shape (see ``kinect.gt.build_shape``).
"""

import numpy as np


def write_off(path, points):
    points = np.asarray(points, dtype=np.float32)
    rows = "\n".join(" ".join(repr(float(c)) for c in p) for p in points)
    with open(path, "w", newline="\n") as f:
        f.write(f"OFF\n# Created by meshio\n\n{len(points)} 0 0\n\n{rows}\n\n")


def read_off(path):
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
    assert lines[0].strip() == "OFF", f"{path} is not an OFF file"
    n = int(lines[1].split()[0])
    return np.array([l.split() for l in lines[2:2 + n]], dtype=np.float64).astype(np.float32)


def write_vts(path, idx):
    with open(path, "w", newline="\n") as f:
        f.write("".join(f"{i}\n" for i in np.asarray(idx, dtype=np.int64)))


def read_vts(path):
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def load_shape(path):
    """Load a shape ``.npz`` into a dict of numpy arrays."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}
