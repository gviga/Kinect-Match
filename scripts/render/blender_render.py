"""Render a scene written by figures.py with BlenderToolbox (Cycles, sun light,
shadow-catching ground, orthographic camera).

    blender --background --python blender_render.py -- scene.npz out.png

Needs BlenderToolbox (https://github.com/HTDerekLiu/BlenderToolbox); set
BLENDER_TOOLBOX to the folder that contains the ``BlenderToolBox`` package.
The scene is already in Blender's z-up frame.
"""

import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.append(os.environ.get("BLENDER_TOOLBOX", ""))
import BlenderToolBox as bt  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
scene_path, out_path = argv[0], os.path.abspath(argv[1])
S = np.load(scene_path, allow_pickle=False)
meta = json.loads(str(S["meta"]))

W, H = meta["res"]
bt.blenderInit(int(W), int(H), int(meta["samples"]), 1.5, True)
scene = bpy.context.scene
# colours are data here: show them as computed (Blender 4's default AgX desaturates)
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.cycles.use_denoising = True

for k, it in enumerate(meta["items"]):
    V, C = S[f"V{k}"], np.clip(S[f"C{k}"], 0, 1)
    if it["kind"] == "points":
        obj = bt.readNumpyPoints(V, (0, 0, 0), (0, 0, 0), (1, 1, 1))
        obj = bt.setPointColors(obj, C)
        bt.setMat_pointCloudColored(obj, bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0), it["radius"])
        continue
    obj = bt.readNumpyMesh(V, S[f"F{k}"].astype(np.int64), (0, 0, 0), (0, 0, 0), (1, 1, 1))
    obj = bt.setMeshColors(obj, C, "vertex")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    bt.setMat_VColor(obj, bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0))

bt.invisibleGround(location=(0, 0, meta["ground"]), shadowBrightness=0.9)

bpy.ops.object.camera_add(location=tuple(meta["cam"]))
cam = bpy.context.object
cam.data.type = "ORTHO"
cam.data.ortho_scale = meta["ortho_scale"]
cam.data.clip_end = 100
back = (Vector(meta["cam"]) - Vector(meta["look"])).normalized()
up = Vector((0, 0, 1))
up = (up - up.dot(back) * back).normalized()
cam.rotation_euler = Matrix((up.cross(back), up, back)).transposed().to_euler()

bpy.ops.object.light_add(type="SUN")
sun = bpy.context.object
sun.rotation_euler = Vector(meta["sun"]).to_track_quat("Z", "Y").to_euler()
sun.data.angle = 0.6
sun.data.energy = 2.2
bt.setLight_ambient(color=(0.2, 0.2, 0.2, 1))
bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")
bt.renderImage(out_path, cam)
