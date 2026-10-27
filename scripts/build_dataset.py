"""Build the KINECT dataset from a local copy of BEHAVE.

    # the 15-shape test set of NAM and FUSE, identical to the one in the papers
    python scripts/build_dataset.py --behave /path/to/behave/sequences --out data/kinect15

    # also the dense pairwise ground truth (level c) between all 210 ordered pairs
    python scripts/build_dataset.py --behave ... --out data/kinect15 --pairs

    # any other BEHAVE frames: a text file with one <sequence>/<frame> per line,
    # or 'all' for every frame with a fit02 registration
    python scripts/build_dataset.py --behave ... --out data/kinect_all --frames all

Output, per shape <name>:
    shapes/<name>.npz      every ground-truth level (kinect.gt.build_shape)
    off_raw/<name>.off     the raw 10k cloud
    off_clean/<name>.off   the cloud after outlier removal
    corres/<name>.vts      for each SMPL vertex, a 0-based index into off_clean
    pairs/<x>__<y>.npz     with --pairs: clean point of x -> clean point of y
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinect import behave, gt, io  # noqa: E402
from kinect.pairs import pair_gt  # noqa: E402


def frames_to_build(args):
    """[(name, frame, fit, subsample indices, scan size, reference .vts)], None where unknown."""
    if args.frames is None:
        split = json.loads(Path(args.split).read_text())
        idx = np.load(ROOT / split["subsample"])
        ref = np.load(ROOT / split["corres"]) if "corres" in split else {}
        return [(s["name"], s["frame"], s["fit"], idx[s["name"]], s["n_scan"], ref.get(s["name"]))
                for s in split["shapes"]]
    if args.frames == "all":
        frames = behave.list_frames(args.behave, args.fit)
    else:
        frames = [l.strip() for l in Path(args.frames).read_text().splitlines() if l.strip()]
    return [(f.replace("/", "_"), f, args.fit, None, None, None) for f in frames]


def use_reference(shape, ref):
    """Swap in the released .vts. It differs from the recomputed one only where
    duplicate points tie (same coordinates), which the check below enforces."""
    clean = shape["points"][shape["inlier"]]
    if not np.array_equal(clean[ref], clean[shape["smpl2clean"]]):
        sys.exit("the rebuilt cloud does not match the released correspondences")
    shape["smpl2clean"] = ref.astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behave", required=True, help="folder holding the BEHAVE sequences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=str(ROOT / "splits/kinect15.json"))
    ap.add_argument("--frames", default=None, help="'all' or a file of <sequence>/<frame> lines (overrides --split)")
    ap.add_argument("--fit", default="fit02")
    ap.add_argument("--n", type=int, default=gt.N_POINTS, help="points per cloud for new frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z", type=float, default=gt.Z_THRESHOLD, help="outlier z-score threshold")
    ap.add_argument("--one-sided", action="store_true", help="keep points unusually close to SMPL")
    ap.add_argument("--tau", type=float, default=gt.TAU, help="visibility / validity radius in metres")
    ap.add_argument("--pairs", action="store_true", help="dense ground truth between all ordered pairs")
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("shapes", "off_raw", "off_clean", "corres"):
        (out / d).mkdir(parents=True, exist_ok=True)

    todo = frames_to_build(args)
    meta = dict(z=args.z, two_sided=not args.one_sided, tau=args.tau, shapes=[])
    for k, (name, frame, fit, idx, n_scan, ref) in enumerate(todo):
        scan, v, f = behave.load_frame(args.behave, frame, fit)
        if n_scan is not None and len(scan) != n_scan:
            sys.exit(f"{frame}: scan has {len(scan)} points, expected {n_scan}. Is this the original BEHAVE release?")
        if idx is None:
            idx = gt.subsample(len(scan), args.n, args.seed + k)
        shape = gt.build_shape(scan, v, f, idx, args.z, not args.one_sided, args.tau)
        if ref is not None and args.z == gt.Z_THRESHOLD and not args.one_sided:
            use_reference(shape, ref)
        np.savez_compressed(out / "shapes" / f"{name}.npz", **shape)
        io.write_off(out / "off_raw" / f"{name}.off", shape["points"])
        io.write_off(out / "off_clean" / f"{name}.off", shape["points"][shape["inlier"]])
        io.write_vts(out / "corres" / f"{name}.vts", shape["smpl2clean"])
        meta["shapes"].append(dict(name=name, frame=frame, fit=fit, n_clean=int(shape["inlier"].sum()),
                                   smpl_visible=float(shape["smpl_visible"].mean())))
        print(f"[{k + 1}/{len(todo)}] {name}  {frame}  clean {shape['inlier'].sum()}/{len(idx)}"
              f"  visible SMPL {shape['smpl_visible'].mean():.0%}", flush=True)

    if args.pairs:
        (out / "pairs").mkdir(exist_ok=True)
        names = [s["name"] for s in meta["shapes"]]
        shapes = {n: io.load_shape(out / "shapes" / f"{n}.npz") for n in names}
        for x, y in itertools.permutations(names, 2):
            p = pair_gt(shapes[x], shapes[y], args.tau)
            np.savez_compressed(out / "pairs" / f"{x}__{y}.npz", **p)
            print(f"pair {x} -> {y}  valid {p['valid'].mean():.0%}", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
