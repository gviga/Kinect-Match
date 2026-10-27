"""Renders of a built dataset: teaser, ground-truth levels, a pair, the gallery.

    BLENDER=/path/to/blender BLENDER_TOOLBOX=/path/to/BlenderToolbox \
        python scripts/render/figures.py --data data/kinect15 --out assets

Colour coding: every SMPL vertex gets a colour from its position on a reference
registration; a point takes the colour of its ground-truth SMPL vertex, so the
same colour means the same place on the body across shapes.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kinect import io  # noqa: E402

HERE = Path(__file__).parent
BLENDER = os.environ.get("BLENDER", "blender")
PELVIS = (1338, 4729)  # front- and back-most SMPL torso vertices at hip height
GREY, RED, ORANGE = (0.78, 0.78, 0.8), (0.80, 0.18, 0.16), (0.93, 0.55, 0.20)
TEASER = ["data_2", "data_118", "data_1011", "data_1283", "data_425", "data_690", "data_812", "data_1353"]


def facing(v):
    d = v[PELVIS[0]] - v[PELVIS[1]]
    d[1] = 0
    return d / np.linalg.norm(d)


def posed(shape, yaw=0.0):
    """SMPL vertices and points, y up, facing +z rotated by ``yaw``, feet on y = 0."""
    flip = np.array([1.0, -1.0, -1.0])  # BEHAVE camera frame (y down) -> y up
    v, p = shape["smpl_v"] * flip, shape["points"] * flip
    a = facing(v)
    t = np.arctan2(a[0], a[2]) - yaw
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    ctr = v.mean(0)
    v, p = (v - ctr) @ R.T, (p - ctr) @ R.T
    off = np.array([v[:, 0].mean(), v[:, 1].min(), v[:, 2].mean()])
    return v - off, p - off


def smpl_colors(ref_v):
    lo, hi = ref_v.min(0), ref_v.max(0)
    return 0.25 + 0.7 * np.clip((ref_v - lo) / (hi - lo), 0, 1)


class Scene:
    def __init__(self):
        self.items = []

    def points(self, P, C, dx=0.0):
        C = np.broadcast_to(np.asarray(C, float), (len(P), 3))
        self.items.append(("points", P + [dx, 0, 0], None, C))

    def mesh(self, V, F, C, dx=0.0):
        C = np.broadcast_to(np.asarray(C, float), (len(V), 3))
        self.items.append(("mesh", V + [dx, 0, 0], F, C))

    def render(self, out, width=2400, samples=96, radius=0.0065, elevation=0.12, margin=0.04):
        zup = lambda x: np.stack([x[..., 0], -x[..., 2], x[..., 1]], -1)  # noqa: E731
        P = np.vstack([it[1] for it in self.items])
        lo, hi = P.min(0), P.max(0)
        c = 0.5 * (lo + hi)
        w = np.array([0.0, elevation, 1.0]) / np.hypot(elevation, 1.0)
        half_w, half_h = (1 + margin) * 0.5 * (hi - lo)[0], (1 + margin) * 0.5 * (hi - lo)[1] + 0.1
        height = int(round(width * half_h / half_w))
        sun = np.array([-0.22, 1.0, 0.28])
        meta = dict(res=[width, height], samples=samples, ortho_scale=float(2 * max(half_w, half_h)),
                    look=list(zup(c)), cam=list(zup(c + 20 * w)), sun=list(zup(sun / np.linalg.norm(sun))),
                    ground=float(lo[1]), items=[])
        arrays = {}
        for k, (kind, V, F, C) in enumerate(self.items):
            arrays[f"V{k}"], arrays[f"C{k}"] = zup(V), np.ascontiguousarray(C)
            if F is not None:
                arrays[f"F{k}"] = F
            meta["items"].append(dict(kind=kind, radius=radius))
        out = Path(out).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.npz"
            np.savez(scene, meta=json.dumps(meta), **arrays)
            r = subprocess.run([BLENDER, "--background", "--python", str(HERE / "blender_render.py"), "--",
                                str(scene), str(out)], capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"Blender failed for {out.name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        crop(out)
        print("wrote", out)


def crop(path, pad=24):
    """Crop to the content and composite on white (renders have a transparent background)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    box = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if box:
        l, t, r, b = box
        im = im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--only", nargs="*", default=None, help="teaser levels pair gallery")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in json.loads((data / "meta.json").read_text())["shapes"]]
    S = {n: io.load_shape(data / "shapes" / f"{n}.npz") for n in names}
    col = smpl_colors(posed(S["data_2"])[0])
    todo = args.only or ["teaser", "levels", "pair", "gallery"]

    def clean(n, yaw=0.0):
        v, p = posed(S[n], yaw)
        m = S[n]["inlier"]
        return v, p[m], col[S[n]["pc2smpl"][m]]

    if "teaser" in todo:
        sc = Scene()
        for k, n in enumerate(TEASER):
            _, p, c = clean(n, yaw=0.35)
            sc.points(p, c, dx=0.85 * k)
        sc.render(out / "teaser.png", width=3000)

    if "levels" in todo:
        n = "data_1283"
        v, p = posed(S[n], 0.35)
        sc = Scene()
        m = S[n]["inlier"]
        sc.points(p, np.where(m[:, None], GREY, RED), dx=0.0)  # (a) raw, outliers in red
        vis = S[n]["smpl_visible"]
        sc.mesh(v, S[n]["smpl_f"], np.where(vis[:, None], GREY, ORANGE), dx=1.0)  # SMPL, unseen in orange
        sc.points(p[m], col[S[n]["pc2smpl"][m]], dx=2.0)  # (b) clean, coloured by correspondence
        sc.render(out / "levels.png", width=2400)

    if "pair" in todo:
        x, y = "data_45", "data_690"
        pair = io.load_shape(data / "pairs" / f"{x}__{y}.npz")
        _, px, _ = clean(x, 0.35)
        _, py, cy = clean(y, 0.35)
        cx = np.where(pair["valid"][:, None], cy[pair["gt"]], RED)  # (c) colour pulled back from y
        sc = Scene()
        sc.points(px, cx, dx=0.0)
        sc.points(py, cy, dx=1.0)
        sc.render(out / "pair.png", width=1600)

    if "gallery" in todo:
        from PIL import Image

        rows = []
        for r, chunk in enumerate((names[:8], names[8:])):
            sc = Scene()
            for k, n in enumerate(chunk):
                _, p, c = clean(n, yaw=0.35)
                sc.points(p, c, dx=0.85 * k)
            sc.render(out / f"_gallery{r}.png", width=3000 * len(chunk) // 8)
            rows.append(Image.open(out / f"_gallery{r}.png"))
        im = Image.new("RGB", (max(i.width for i in rows), sum(i.height for i in rows)), "white")
        im.paste(rows[0], (0, 0))
        im.paste(rows[1], ((im.width - rows[1].width) // 2, rows[0].height))
        im.save(out / "gallery.png")
        for r in range(2):
            (out / f"_gallery{r}.png").unlink()


if __name__ == "__main__":
    main()
