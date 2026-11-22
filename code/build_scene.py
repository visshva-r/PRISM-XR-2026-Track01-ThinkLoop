"""Build scene.blend from the ICT-FaceKit GNM head + synthesised UV textures.

Run:  blender -b --python code/build_scene.py
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import ict_obj  # noqa: E402

ROOT = ict_obj.ROOT
TEX = os.path.join(ROOT, "textures")

EXPRESSION_SHAPES = [
    "eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R",
    "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
    "eyeWide_L", "eyeWide_R", "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R",
    "browOuterUp_L", "browOuterUp_R", "cheekSquint_L", "cheekSquint_R", "mouthSmile_L", "mouthSmile_R",
]
IDENTITY_SHAPES = [f"identity{i:03d}" for i in range(5)]

FIXATION_DIST = 0.35
CAMERA_DIST = 0.40
DEFAULT_VARIANT = 4


def reset_file():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_mesh(name, V_bl, sub_faces, VT):
    vids = sorted({i for _, vi, _ in sub_faces for i in vi})
    remap = {old: new for new, old in enumerate(vids)}
    polys = [[remap[i] for i in vi] for _, vi, _ in sub_faces]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V_bl[vids].tolist(), [], polys)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    loop_uv = np.array([VT[t] for _, _, ti in sub_faces for t in ti], np.float32).ravel()
    uv.data.foreach_set("uv", loop_uv)
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj, np.array(vids)


def assign_materials(obj, sub_faces, mat_map):
    """mat_map: ICT material name -> bpy material."""
    order = []
    for m in mat_map.values():
        if m not in order:
            order.append(m)
    for m in order:
        obj.data.materials.append(m)
    idx = [order.index(mat_map[mat]) for mat, _, _ in sub_faces]
    obj.data.polygons.foreach_set("material_index", idx)


def add_shape_keys(obj, vids, names):
    obj.shape_key_add(name="Basis", from_mix=False)
    for n in names:
        P = ict_obj.load_verts(os.path.join(ict_obj.ASSETS, n + ".obj"))
        co = ict_obj.to_blender(P[vids]).astype(np.float32).ravel()
        sk = obj.shape_key_add(name=n, from_mix=False)
        sk.data.foreach_set("co", co)
        sk.value = 0.0
        sk.slider_min = 0.0
        sk.slider_max = 1.0


# ------------------------------------------------------------------ materials


def _nodes(mat):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)
    return nt, out


def _img(nt, path, noncolor=False, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(path, check_existing=True)
    if noncolor:
        img.colorspace_settings.name = "Non-Color"
    n.image = img
    n.location = loc
    return n


def _set(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def skin_material(variant=1):
    mat = bpy.data.materials.new("M_Skin")
    nt, out = _nodes(mat)
    L = nt.links
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(uvn.outputs["UV"], sep.inputs[0])
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 1.0
    L.new(sep.outputs["X"], gt.inputs[0])
    shift = nt.nodes.new("ShaderNodeVectorMath")
    shift.operation = "SUBTRACT"
    shift.inputs[1].default_value = (1.0, 0.0, 0.0)
    L.new(uvn.outputs["UV"], shift.inputs[0])

    vdir = os.path.join(TEX, f"variant_{variant:02d}")
    pairs = {
        "color": (os.path.join(vdir, "face_1001.png"), os.path.join(vdir, "head_1002.png"), False),
        "rough": (os.path.join(TEX, "shared", "roughness_1001.png"), os.path.join(TEX, "shared", "roughness_1002.png"), True),
        "height": (os.path.join(TEX, "shared", "height_1001.png"), os.path.join(TEX, "shared", "height_1002.png"), True),
    }
    mixed = {}
    for k, (a, b, nc) in pairs.items():
        ia = _img(nt, a, nc)
        ib = _img(nt, b, nc)
        ia.name, ib.name = f"skin_{k}_1001", f"skin_{k}_1002"
        ia.image.name, ib.image.name = f"skin_{k}_1001", f"skin_{k}_1002"
        L.new(uvn.outputs["UV"], ia.inputs["Vector"])
        L.new(shift.outputs["Vector"], ib.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        L.new(gt.outputs[0], mix.inputs["Factor"])
        L.new(ia.outputs["Color"], mix.inputs[6])
        L.new(ib.outputs["Color"], mix.inputs[7])
        mixed[k] = mix.outputs[2]

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    L.new(mixed["color"], bsdf.inputs["Base Color"])
    L.new(mixed["rough"], bsdf.inputs["Roughness"])
    _set(bsdf, "Subsurface Weight", 0.3)
    _set(bsdf, "Subsurface Radius", (3.67, 1.37, 0.68))
    _set(bsdf, "Subsurface Scale", 0.0008)
    _set(bsdf, "Specular IOR Level", 0.45)
    _set(bsdf, "Coat Weight", 0.04)
    _set(bsdf, "Coat Roughness", 0.35)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0004
    L.new(mixed["height"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def eye_material(side, variant=1):
    mat = bpy.data.materials.new(f"M_Eye_{side}")
    nt, out = _nodes(mat)
    L = nt.links
    tex = _img(nt, os.path.join(TEX, f"variant_{variant:02d}", f"eye_{side}.png"))
    tex.name = tex.image.name = f"eye_{side}"
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.4)
    _set(bsdf, "Subsurface Weight", 0.15)
    _set(bsdf, "Subsurface Radius", (1.0, 0.6, 0.5))
    _set(bsdf, "Subsurface Scale", 0.001)
    _set(bsdf, "Specular IOR Level", 0.2)
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def wet_material(name, base_alpha=0.05, rough=0.02):
    """Clear tear-film / cornea: transparent with a sharp Fresnel specular highlight."""
    mat = bpy.data.materials.new(name)
    nt, out = _nodes(mat)
    L = nt.links
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    gloss = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set(gloss, "Base Color", (0.0, 0.0, 0.0, 1.0))
    _set(gloss, "Roughness", rough)
    _set(gloss, "Specular IOR Level", 1.0)
    _set(gloss, "IOR", 1.376)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.use_clamp = True
    add.inputs[1].default_value = base_alpha
    L.new(lw.outputs["Fresnel"], add.inputs[0])
    mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(add.outputs[0], mix.inputs[0])
    L.new(transp.outputs[0], mix.inputs[1])
    L.new(gloss.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"), ("use_backface_culling", True)):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass
    return mat


def lash_material(eye_z):
    """Eyelash cards: strand-like alpha from vertically stretched noise, so lashes read as hairs, not a band.
    Lower lashes (below the eye centre) are kept sparser, as in real eyes."""
    mat = bpy.data.materials.new("M_Lashes")
    nt, out = _nodes(mat)
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (5200.0, 5200.0, 140.0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.0
    nz.inputs["Detail"].default_value = 0.0
    L.new(mp.outputs["Vector"], nz.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.60
    ramp.color_ramp.elements[1].position = 0.66
    L.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    upper = nt.nodes.new("ShaderNodeMapRange")
    upper.inputs["From Min"].default_value = eye_z - 0.004
    upper.inputs["From Max"].default_value = eye_z
    upper.inputs["To Min"].default_value = 0.3
    upper.inputs["To Max"].default_value = 1.0
    L.new(sep.outputs["Z"], upper.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L.new(ramp.outputs["Color"], mul.inputs[0])
    L.new(upper.outputs["Result"], mul.inputs[1])
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set(bsdf, "Base Color", (0.02, 0.015, 0.012, 1.0))
    _set(bsdf, "Roughness", 0.45)
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(mul.outputs[0], mix.inputs[0])
    L.new(transp.outputs[0], mix.inputs[1])
    L.new(bsdf.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def simple_material(name, color, rough=0.5, alpha=1.0, sss=0.0):
    mat = bpy.data.materials.new(name)
    nt, out = _nodes(mat)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Roughness", rough)
    _set(bsdf, "Alpha", alpha)
    _set(bsdf, "Subsurface Weight", sss)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# ------------------------------------------------------------------ scene


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_area(name, loc, target, power, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def setup_world(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg_env = nt.nodes.new("ShaderNodeBackground")
    bg_cam = nt.nodes.new("ShaderNodeBackground")
    bg_cam.inputs["Color"].default_value = (0.020, 0.022, 0.026, 1)
    bg_cam.inputs["Strength"].default_value = 1.0
    sl = os.path.join(bpy.utils.system_resource("DATAFILES"), "studiolights", "world")
    hdr = None
    if os.path.isdir(sl):
        for pref in ("interior.exr", "studio.exr", "city.exr", "forest.exr"):
            if os.path.exists(os.path.join(sl, pref)):
                hdr = os.path.join(sl, pref)
                break
    if hdr:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdr, check_existing=True)
        nt.links.new(env.outputs["Color"], bg_env.inputs["Color"])
    bg_env.inputs["Strength"].default_value = 0.3
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg_env.outputs[0], mix.inputs[1])
    nt.links.new(bg_cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def setup_render(scene):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    ee = scene.eevee
    for attr, val in (("taa_render_samples", 64), ("use_raytracing", True), ("use_shadows", True),
                      ("shadow_ray_count", 2), ("shadow_step_count", 8), ("use_gtao", True)):
        try:
            setattr(ee, attr, val)
        except Exception:
            pass
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.exposure = -0.55
        for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
            try:
                scene.view_settings.look = look
                break
            except TypeError:
                continue
    except Exception:
        pass


def main():
    reset_file()
    scene = bpy.context.scene
    V, VT, faces = ict_obj.load_obj()
    V_bl = ict_obj.to_blender(V)
    centers = ict_obj.eye_centers(V)

    head_faces = [f for f in faces if f[0] not in ict_obj.EYE_MATERIALS | ict_obj.DROPPED_MATERIALS]
    head, head_vids = make_mesh("GNM_Head", V_bl, head_faces, VT)
    skin = skin_material(DEFAULT_VARIANT)
    assign_materials(head, head_faces, {
        "M_Face": skin, "M_BackHead": skin,
        "M_GumsTongue": simple_material("M_Gums", (0.45, 0.12, 0.13), 0.35, sss=0.3),
        "M_Teeth": simple_material("M_Teeth", (0.86, 0.82, 0.74), 0.25),
        "M_LacrimalFluid": wet_material("M_TearFilm", 0.0, 0.03),
        "M_EyeLashes": lash_material(float(ict_obj.to_blender(centers["L"][0])[2])),
    })
    add_shape_keys(head, head_vids, EXPRESSION_SHAPES + IDENTITY_SHAPES)

    # Armature with one bone per eye, pivoting at the fitted eyeball centre.
    arm_data = bpy.data.armatures.new("EyeRig")
    rig = bpy.data.objects.new("EyeRig", arm_data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eye_world = {}
    for side in ("L", "R"):
        c = Vector(ict_obj.to_blender(centers[side][0]).tolist())
        eye_world[side] = c
        b = arm_data.edit_bones.new(f"eye.{side}")
        b.head = c
        b.tail = c + Vector((0.0, -0.02, 0.0))
        b.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")

    mid = (eye_world["L"] + eye_world["R"]) / 2
    target = bpy.data.objects.new("GazeTarget", None)
    target.empty_display_type = "SPHERE"
    target.empty_display_size = 0.01
    target.location = mid + Vector((0.0, -FIXATION_DIST, 0.0))
    scene.collection.objects.link(target)

    cornea = wet_material("M_Cornea", 0.02, 0.015)
    for side in ("L", "R"):
        mats = {"M_ScleraLeft", "M_IrisLeft"} if side == "L" else {"M_ScleraRight", "M_IrisRight"}
        ef = [f for f in faces if f[0] in mats]
        eye, _ = make_mesh(f"Eye_{side}", V_bl, ef, VT)
        inner = eye_material(side, DEFAULT_VARIANT)
        outer_name = "M_ScleraLeft" if side == "L" else "M_ScleraRight"
        inner_name = "M_IrisLeft" if side == "L" else "M_IrisRight"
        assign_materials(eye, ef, {outer_name: cornea, inner_name: inner})
        vg = eye.vertex_groups.new(name=f"eye.{side}")
        vg.add(list(range(len(eye.data.vertices))), 1.0, "REPLACE")
        mod = eye.modifiers.new("EyeRig", "ARMATURE")
        mod.object = rig
        pb = rig.pose.bones[f"eye.{side}"]
        con = pb.constraints.new("DAMPED_TRACK")
        con.target = target
        con.track_axis = "TRACK_Y"

    # identity eye-centre offsets so the eyeballs follow identity reshaping
    neutral_c = {s: ict_obj.to_blender(centers[s][0]) for s in ("L", "R")}
    id_offsets = {}
    for n in IDENTITY_SHAPES:
        Pi = ict_obj.load_verts(os.path.join(ict_obj.ASSETS, n + ".obj"))
        ci = ict_obj.eye_centers(Pi)
        id_offsets[n] = {s: (ict_obj.to_blender(ci[s][0]) - neutral_c[s]).tolist() for s in ("L", "R")}

    # cameras: locked 40 cm in front of the eye midpoint
    cam_d = bpy.data.cameras.new("Cam_Main")
    cam_d.lens = 45.0
    cam_d.sensor_width = 36.0
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.shift_y = -0.12
    cam_d.clip_start = 0.01
    cam = bpy.data.objects.new("Cam_Main", cam_d)
    scene.collection.objects.link(cam)
    cam.location = mid + Vector((0.0, -CAMERA_DIST, 0.0))
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.camera = cam

    close_d = bpy.data.cameras.new("Cam_Eyes")
    close_d.lens = 120.0
    close_d.sensor_width = 36.0
    close_d.sensor_fit = "HORIZONTAL"
    close_d.clip_start = 0.01
    close = bpy.data.objects.new("Cam_Eyes", close_d)
    scene.collection.objects.link(close)
    close.location = cam.location.copy()
    close.rotation_euler = cam.rotation_euler.copy()

    tq_d = bpy.data.cameras.new("Cam_ThreeQuarter")
    tq_d.lens = 45.0
    tq_d.sensor_width = 36.0
    tq_d.clip_start = 0.01
    tq = bpy.data.objects.new("Cam_ThreeQuarter", tq_d)
    scene.collection.objects.link(tq)
    ang = math.radians(32)
    head_c = mid + Vector((0.0, 0.05, -0.04))
    tq.location = head_c + Vector((0.55 * math.sin(ang), -0.55 * math.cos(ang), 0.03))
    look_at(tq, head_c)

    # portrait lighting: soft key, fill, rim + dim environment
    hc = mid + Vector((0, 0.04, -0.03))
    add_area("Key", hc + Vector((-0.55, -0.75, 0.45)), hc, 40.0, 0.7, (1.0, 0.97, 0.93))
    add_area("Fill", hc + Vector((0.8, -0.55, 0.05)), hc, 8.0, 1.0, (0.93, 0.96, 1.0))
    add_area("Rim", hc + Vector((0.45, 0.6, 0.5)), hc, 35.0, 0.6, (1.0, 0.98, 0.95))
    add_area("Catch", mid + Vector((-0.12, -0.9, 0.22)), mid, 4.0, 0.08)
    setup_world(scene)
    setup_render(scene)

    cfg = {
        "eye_center_L": list(eye_world["L"]), "eye_center_R": list(eye_world["R"]),
        "eye_midpoint": list(mid), "eye_radius": centers["L"][1] * 0.01,
        "fixation_distance": FIXATION_DIST, "camera_distance": CAMERA_DIST,
        "identity_eye_offsets": id_offsets, "texture_variant": DEFAULT_VARIANT,
    }
    scene["eyes_config"] = json.dumps(cfg)

    # Bake the "Generate Motion" expression into the Timeline so the .blend plays it back directly.
    sys.path.insert(0, os.path.join(ROOT, "wrapper"))
    import eyes as eyes_mod  # noqa: E402
    rig_api = eyes_mod.Eyes()
    rig_api.keyframe_sequence("motion")
    scene.frame_set(1)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "scene.blend"), relative_remap=True, compress=True)
    print("SAVED", os.path.join(ROOT, "scene.blend"))


if __name__ == "__main__":
    main()
