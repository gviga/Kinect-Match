"""Evaluate point-to-point maps on KINECT.

    python scripts/evaluate.py --data data/kinect15 --maps my_maps/ [--out results.json]

my_maps/ holds one file per ordered pair, ``<x>__<y>.txt`` or ``.npy``
(e.g. ``data_1011__data_118.txt``): one integer per line, line i = 0-based
index in off_clean/<y>.off of the image of point i of off_clean/<x>.off.

Reported, averaged over the pairs found (see kinect.metrics.evaluate):
  geo / euc    mean geodesic error (/ geodesic diameter of the target SMPL) and
               mean Euclidean error (metres)
  smpl         on the 6890 SMPL vertices (.vts), the NAM / FUSE protocol
  smpl_visible same, only on SMPL vertices seen in both clouds
  dense        on every clean point (needs pairs/, from build_dataset.py --pairs)
  dense_valid  same, only where the pairwise ground truth is valid
  coverage     fraction of the target cloud hit by the map

``--maps gt`` evaluates the pairwise ground truth itself, as a sanity check.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinect import io  # noqa: E402
from kinect.metrics import Geodesics, evaluate  # noqa: E402

LEVELS = ("smpl", "smpl_visible", "dense", "dense_valid")


def load_map(maps, x, y):
    for ext in (".npy", ".txt"):
        p = Path(maps) / f"{x}__{y}{ext}"
        if p.exists():
            return np.load(p) if ext == ".npy" else np.loadtxt(p, dtype=np.int64)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--maps", required=True, help="folder of maps, or 'gt'")
    ap.add_argument("--out", default=None, help="write per-pair results to this JSON")
    ap.add_argument("--no-geodesic", action="store_true", help="Euclidean errors only (faster)")
    args = ap.parse_args()

    data = Path(args.data)
    names = [s["name"] for s in json.loads((data / "meta.json").read_text())["shapes"]]
    shapes = {n: io.load_shape(data / "shapes" / f"{n}.npz") for n in names}

    results = {}
    for y in names:  # grouped by target: one geodesic matrix at a time
        geo = None
        for x in names:
            if x == y:
                continue
            pair_path = data / "pairs" / f"{x}__{y}.npz"
            pair = io.load_shape(pair_path) if pair_path.exists() else None
            T = pair["gt"] if args.maps == "gt" and pair is not None else load_map(args.maps, x, y)
            if T is None:
                continue
            if geo is None and not args.no_geodesic:
                geo = Geodesics(shapes[y])
            results[f"{x}__{y}"] = r = evaluate(T, shapes[x], shapes[y], pair, geo)
            print(f"{x} -> {y}  " + "  ".join(f"{l} {r[l].get('geo', r[l]['euc']):.4f}" for l in LEVELS if l in r),
                  flush=True)
    if not results:
        sys.exit("no maps found")

    mean = defaultdict(list)
    for r in results.values():
        mean["coverage"].append(r["coverage"])
        for l in LEVELS:
            for k in ("geo", "euc"):
                if l in r and k in r[l]:
                    mean[f"{l}/{k}"].append(r[l][k])
    summary = {k: float(np.mean(v)) for k, v in mean.items()}
    print(f"\n{len(results)} pairs")
    for k, v in summary.items():
        print(f"  {k:<18} {v:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(summary=summary, pairs=results), indent=1))


if __name__ == "__main__":
    main()
