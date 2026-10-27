"""Check a build of the test set against the files used in NAM and FUSE.

    python scripts/verify.py data/kinect15

Compares the SHA-256 of every off_clean/*.off and corres/*.vts with
splits/kinect15_sha256.json, computed on the original files.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = Path(sys.argv[1])
    ref = json.loads((ROOT / "splits/kinect15_sha256.json").read_text())
    bad = 0
    for rel, h in ref.items():
        p = data / rel
        ok = p.exists() and sha256(p) == h
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'}  {rel}")
    print(f"\n{len(ref) - bad}/{len(ref)} files identical to the released test set")
    sys.exit(bad > 0)


if __name__ == "__main__":
    main()
