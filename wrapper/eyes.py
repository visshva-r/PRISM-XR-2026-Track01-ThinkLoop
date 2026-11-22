"""Synthetic eye-movement control wrapper for the GNM head (scene.blend).

Python API (inside Blender):
    import eyes
    eyes.move(h, v)        # h, v in [-1, 1]; +h = image right, +v = up
    eyes.reset()           # centre gaze, eyelids open
    eyes.blink(1.0)        # 0 open .. 1 closed
    eyes.render("out.png") # 1024x1024 PNG from the locked 40 cm camera

CLI (headless):
    blender -b scene.blend --python wrapper/eyes.py -- --input left
    blender -b scene.blend --python wrapper/eyes.py -- --input right
    blender -b scene.blend --python wrapper/eyes.py -- --input motion
    blender -b scene.blend --python wrapper/eyes.py -- --input clockwise   # 10 renders + metadata.json
    blender -b scene.blend --python wrapper/eyes.py -- --input directions  # L/R/U/D/reset pass-fail check
    blender -b scene.blend --python wrapper/eyes.py -- --input textures    # one render per UV texture variant
    blender -b scene.blend --python wrapper/eyes.py -- --input all

Gaze model: both eye bones Damped-Track a single fixation target placed 35 cm in
front of the eye midpoint, so the eyeballs rotate spherically about their own
centres and converge (vergence) on the target. move(h, v) slides the target over
a plane at 35 cm (+-20 cm horizontally, +-15 cm vertically, i.e. ~30 / ~23 deg).
Eyelids, brows and lid-gaze coupling are driven by GNM blendshapes.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANGE_H = 0.20
RANGE_V = 0.15
FPS = 30

MANAGED = [
    "eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R",
    "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
    "eyeWide_L", "eyeWide_R", "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R",
    "browOuterUp_L", "browOuterUp_R", "cheekSquint_L", "cheekSquint_R", "mouthSmile_L", "mouthSmile_R",
]
EXPRESSIONS = ("wide", "squint", "brow_up", "brow_down", "smile")


class Eyes:
    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        self.cfg = json.loads(self.scene["eyes_config"])
        self.target = bpy.data.objects["GazeTarget"]
        self.rig = bpy.data.objects["EyeRig"]
        self.head = bpy.data.objects["GNM_Head"]
        self.keys = self.head.data.shape_keys.key_blocks
        self.mid = Vector(self.cfg["eye_midpoint"])
        self.h = self.v = 0.0
        self.blink_amt = 0.0
        self.expr = {k: 0.0 for k in EXPRESSIONS}

    # ---------------------------------------------------------- controls
    def move(self, h, v, frame=None):
        self.h = float(max(-1.0, min(1.0, h)))
        self.v = float(max(-1.0, min(1.0, v)))
        return self._apply(frame)

    def reset(self, frame=None):
        self.blink_amt = 0.0
        self.expr = {k: 0.0 for k in EXPRESSIONS}
        return self.move(0.0, 0.0, frame)

    def blink(self, amount=1.0, frame=None):
        self.blink_amt = float(max(0.0, min(1.0, amount)))
        return self._apply(frame)

    def expression(self, frame=None, **kw):
        for k, val in kw.items():
            if k in self.expr:
                self.expr[k] = float(val)
        return self._apply(frame)

    def set_identity(self, weights):
        """weights: dict identityNNN -> [0,1]. Eye pivots follow the reshaped sockets."""
        off = {"L": Vector(), "R": Vector()}
        for n, sk in self.keys.items():
            if n.startswith("identity"):
                w = float(weights.get(n, 0.0))
                sk.value = w
                for s in ("L", "R"):
                    off[s] += Vector(self.cfg["identity_eye_offsets"][n][s]) * w
        for s in ("L", "R"):
            pb = self.rig.pose.bones[f"eye.{s}"]
            pb.location = pb.bone.matrix_local.to_3x3().inverted() @ off[s]
        self.mid = Vector(self.cfg["eye_midpoint"]) + (off["L"] + off["R"]) / 2
        return self._apply(None)

    def set_texture(self, variant):
        d = os.path.join(ROOT, "textures", f"variant_{int(variant):02d}")
        for name, fn in (("skin_color_1001", "face_1001.png"), ("skin_color_1002", "head_1002.png"),
                         ("eye_L", "eye_L.png"), ("eye_R", "eye_R.png")):
            img = bpy.data.images.get(name)
            if img:
                img.filepath = os.path.join(d, fn)
                img.reload()
        self.cfg["texture_variant"] = int(variant)

    # ---------------------------------------------------------- internals
    def _shape_values(self):
        h, v, b = self.h, self.v, self.blink_amt
        vals = {k: 0.0 for k in MANAGED}
        if v < 0:
            vals["eyeLookDown_L"] = vals["eyeLookDown_R"] = min(1.0, -v * 0.55)
        else:
            vals["eyeLookUp_L"] = vals["eyeLookUp_R"] = v * 0.7
            vals["browInnerUp_L"] = vals["browInnerUp_R"] = v * 0.22
            vals["browOuterUp_L"] = vals["browOuterUp_R"] = v * 0.18
        if h < 0:  # image-left = subject's right
            vals["eyeLookOut_R"] = vals["eyeLookIn_L"] = -h * 0.55
        else:
            vals["eyeLookOut_L"] = vals["eyeLookIn_R"] = h * 0.55
        e = self.expr
        for s in ("L", "R"):
            vals[f"eyeWide_{s}"] = e["wide"]
            vals[f"eyeSquint_{s}"] = max(e["squint"], 0.0)
            vals[f"cheekSquint_{s}"] = e["squint"] * 0.8 + e["smile"] * 0.35
            vals[f"mouthSmile_{s}"] = e["smile"]
            vals[f"browInnerUp_{s}"] = min(1.0, vals[f"browInnerUp_{s}"] + e["brow_up"])
            vals[f"browOuterUp_{s}"] = min(1.0, vals[f"browOuterUp_{s}"] + e["brow_up"] * 0.85)
            vals[f"browDown_{s}"] = e["brow_down"]
            vals[f"eyeBlink_{s}"] = b
            for k in (f"eyeLookUp_{s}", f"eyeLookDown_{s}", f"eyeWide_{s}", f"eyeSquint_{s}"):
                vals[k] *= (1.0 - b)
        return vals

    def _apply(self, frame):
        # Bell's phenomenon: eyes drift slightly down/in while the lids are closed
        hv = self.h * (1 - 0.15 * self.blink_amt)
        vv = self.v - 0.12 * self.blink_amt
        self.target.location = self.mid + Vector((hv * RANGE_H, -self.cfg["fixation_distance"], vv * RANGE_V))
        for k, val in self._shape_values().items():
            sk = self.keys.get(k)
            if sk is not None:
                sk.value = val
                if frame is not None:
                    sk.keyframe_insert("value", frame=frame)
        if frame is not None:
            self.target.keyframe_insert("location", frame=frame)
        return self

    # ---------------------------------------------------------- measurement
    def camera_matrix(self, cam=None):
        cam = cam or self.scene.camera
        r = self.scene.render
        W = r.resolution_x * r.resolution_percentage / 100.0
        H = r.resolution_y * r.resolution_percentage / 100.0
        cd = cam.data
        fx = cd.lens / cd.sensor_width * W
        K = [[fx, 0.0, W * (0.5 - cd.shift_x)], [0.0, fx, H * 0.5 + cd.shift_y * max(W, H)], [0.0, 0.0, 1.0]]
        R_bc = cam.matrix_world.to_3x3().transposed()
        t_bc = -(R_bc @ cam.matrix_world.translation)
        flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        R = flip @ R_bc
        t = flip @ t_bc
        return {"K": K, "R": [list(row) for row in R], "t": list(t),
                "P": (np.array(K) @ np.c_[np.array([list(row) for row in R]), np.array(list(t))]).tolist(),
                "convention": "OpenCV pinhole, world->camera, pixels with origin top-left",
                "width": int(W), "height": int(H), "lens_mm": cd.lens, "sensor_mm": cd.sensor_width}

    def state(self, cam=None):
        bpy.context.view_layer.update()
        cam = cam or self.scene.camera
        r = self.scene.render
        W = r.resolution_x * r.resolution_percentage / 100.0
        H = r.resolution_y * r.resolution_percentage / 100.0
        out = {"input": {"h": round(self.h, 4), "v": round(self.v, 4)},
               "fixation_target": [round(c, 5) for c in self.target.matrix_world.translation]}
        for s in ("L", "R"):
            pb = self.rig.pose.bones[f"eye.{s}"]
            Mw = self.rig.matrix_world @ pb.matrix
            Mr = self.rig.matrix_world @ pb.bone.matrix_local
            Rd = Mw.to_3x3() @ Mr.to_3x3().inverted()
            e = Rd.to_euler("XYZ")
            gaze = (Rd @ Vector((0.0, -1.0, 0.0))).normalized()
            centre = Mw.translation
            pupil = centre + gaze * self.cfg["eye_radius"]
            p = world_to_camera_view(self.scene, cam, pupil)
            out[f"eye_rotation_{s}"] = [round(a, 6) for a in e]
            out[f"eye_rotation_{s}_deg"] = [round(math.degrees(a), 3) for a in e]
            out[f"gaze_dir_{s}"] = [round(c, 6) for c in gaze]
            out[f"eye_center_3D_{s}"] = [round(c, 6) for c in centre]
            out[f"pupil_center_3D_{s}"] = [round(c, 6) for c in pupil]
            out[f"pupil_center_2D_{s}"] = [round(p.x * W, 2), round((1.0 - p.y) * H, 2)]
        gl, gr = Vector(out["gaze_dir_L"]), Vector(out["gaze_dir_R"])
        out["vergence_deg"] = round(math.degrees(gl.angle(gr)), 3)
        return out

    # ---------------------------------------------------------- rendering
    def render(self, path, camera=None, samples=None):
        sc = self.scene
        if camera:
            sc.camera = bpy.data.objects[camera]
        if samples:
            try:
                sc.eevee.taa_render_samples = samples
            except Exception:
                pass
        _set_image_output(sc)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sc.render.filepath = os.path.abspath(path)
        bpy.ops.render.render(write_still=True)
        return path

    # ---------------------------------------------------------- sequences
    def clear_animation(self):
        self.target.animation_data_clear()
        if self.head.data.shape_keys.animation_data:
            self.head.data.shape_keys.animation_data_clear()

    def keyframe_sequence(self, name):
        tl = SEQUENCES[name]()
        self.clear_animation()
        n = len(tl["h"])
        for i in range(n):
            self.h, self.v = float(tl["h"][i]), float(tl["v"][i])
            self.blink_amt = float(tl["blink"][i])
            for k in EXPRESSIONS:
                self.expr[k] = float(tl[k][i])
            self._apply(i + 1)
        self.scene.frame_start = 1
        self.scene.frame_end = n
        self.scene.render.fps = FPS
        return n

    def render_sequence(self, name, path, samples=32):
        n = self.keyframe_sequence(name)
        sc = self.scene
        sc.camera = bpy.data.objects["Cam_Main"]
        try:
            sc.eevee.taa_render_samples = samples
        except Exception:
            pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if _set_video_output(sc):
            sc.render.filepath = os.path.abspath(path)
            bpy.ops.render.render(animation=True)
        else:
            frames = os.path.splitext(os.path.abspath(path))[0] + "_frames"
            _set_image_output(sc)
            sc.render.filepath = os.path.join(frames, "frame_")
            bpy.ops.render.render(animation=True)
        return n


def _set_image_output(sc):
    im = sc.render.image_settings
    if hasattr(im, "media_type"):
        try:
            im.media_type = "IMAGE"
        except Exception:
            pass
    im.file_format = "PNG"
    im.color_mode = "RGB"


def _set_video_output(sc):
    im = sc.render.image_settings
    try:
        if hasattr(im, "media_type"):
            im.media_type = "VIDEO"
        im.file_format = "FFMPEG"
        ff = sc.render.ffmpeg
        ff.format = "MPEG4"
        ff.codec = "H264"
        ff.constant_rate_factor = "HIGH"
        ff.ffmpeg_preset = "GOOD"
        ff.gopsize = 15
        return True
    except Exception as e:  # pragma: no cover
        print("video output unavailable, writing PNG frames:", e)
        return False


# ================================================================ motion synthesis


def _min_jerk(t):
    t = np.clip(t, 0, 1)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


class Timeline:
    """Per-frame channels built from human eye-movement primitives."""

    def __init__(self, seconds, seed=0):
        self.n = int(round(seconds * FPS))
        self.t = np.arange(self.n) / FPS
        self.h = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.blink = np.zeros(self.n)
        self.ch = {k: np.zeros(self.n) for k in EXPRESSIONS}
        self._gaze = [(0.0, 0.0, 0.0)]  # (time, h, v)
        self._pursuits = []
        self.rng = np.random.default_rng(seed)

    def saccade(self, t, h, v):
        self._gaze.append((t, h, v))
        return self

    def pursuit(self, t0, dur, fn):
        self._pursuits.append((t0, dur, fn))
        return self

    def add_blink(self, t, amp=1.0, close=0.085, hold=0.04, open_=0.17):
        tc = self.t - t
        b = np.where(tc < 0, 0.0, np.where(tc < close, _min_jerk(tc / close),
                     np.where(tc < close + hold, 1.0, 1.0 - _min_jerk((tc - close - hold) / open_))))
        b = np.clip(b, 0, 1) * amp
        self.blink = np.maximum(self.blink, b)
        return self

    def expr(self, name, keys):
        """keys: [(time, value), ...] eased with minimum-jerk between keys."""
        out = np.zeros(self.n)
        keys = sorted(keys)
        for i, tt in enumerate(self.t):
            if tt <= keys[0][0]:
                out[i] = keys[0][1]
            elif tt >= keys[-1][0]:
                out[i] = keys[-1][1]
            else:
                for (t0, a), (t1, b) in zip(keys, keys[1:]):
                    if t0 <= tt <= t1:
                        out[i] = a + (b - a) * _min_jerk((tt - t0) / max(t1 - t0, 1e-6))
                        break
        self.ch[name] = out
        return self

    def build(self):
        g = sorted(self._gaze)
        h = np.zeros(self.n)
        v = np.zeros(self.n)
        cur_h, cur_v = g[0][1], g[0][2]
        h[:] = cur_h
        v[:] = cur_v
        for (t0, h0, v0), (t1, h1, v1) in zip(g, g[1:]):
            amp = math.degrees(math.atan(math.hypot((h1 - h0) * RANGE_H, (v1 - v0) * RANGE_V) / 0.35))
            dur = 0.021 + 0.0022 * amp  # saccade main sequence
            m = self.t >= t1
            s = _min_jerk((self.t - t1) / dur)
            # slight overshoot + glissade, typical of large saccades
            over = 0.06 * np.exp(-((self.t - t1 - dur) / 0.03) ** 2) * (amp > 12)
            h[m] = h0 + (h1 - h0) * (s[m] + over[m])
            v[m] = v0 + (v1 - v0) * (s[m] + over[m])
        for t0, dur, fn in self._pursuits:
            m = (self.t >= t0) & (self.t <= t0 + dur)
            for i in np.where(m)[0]:
                h[i], v[i] = fn((self.t[i] - t0) / dur)
        # fixational drift + tremor
        ph = self.rng.uniform(0, 2 * np.pi, 6)
        drift_h = 0.006 * np.sin(2 * np.pi * 0.7 * self.t + ph[0]) + 0.004 * np.sin(2 * np.pi * 1.9 * self.t + ph[1])
        drift_v = 0.006 * np.sin(2 * np.pi * 0.55 * self.t + ph[2]) + 0.004 * np.sin(2 * np.pi * 2.3 * self.t + ph[3])
        self.h = np.clip(h + drift_h, -1, 1)
        self.v = np.clip(v + drift_v, -1, 1)
        return {"h": self.h, "v": self.v, "blink": self.blink, **self.ch}


def seq_side(sign):
    def make():
        tl = Timeline(7.0, seed=3 if sign < 0 else 4)
        tl.saccade(0.8, 0.85 * sign, 0.02).saccade(1.05, 0.9 * sign, -0.03)
        tl.saccade(2.9, 0.0, 0.0)
        tl.saccade(4.2, 0.55 * sign, 0.1).saccade(5.6, 0.0, 0.0)
        tl.add_blink(0.25, 0.9).add_blink(2.0).add_blink(4.85).add_blink(6.4, 0.95)
        tl.expr("squint", [(0, 0), (1.0, 0.0), (1.4, 0.08), (2.8, 0.08), (3.1, 0.0)])
        return tl.build()
    return make


def seq_motion():
    tl = Timeline(12.0, seed=7)
    tl.saccade(0.6, -0.8, 0.0).saccade(1.4, 0.8, 0.05).saccade(2.3, -0.4, 0.7).saccade(2.9, 0.0, 0.85)
    # eye roll: smooth circular pursuit up -> left -> down -> right -> up
    tl.pursuit(3.15, 2.2, lambda u: (-0.8 * math.sin(2 * math.pi * u), 0.85 * math.cos(2 * math.pi * u)))
    tl.saccade(3.15, 0.0, 0.85)
    tl.saccade(5.6, 0.0, 0.0)
    # reading: small rightward saccades then a return sweep
    for t, h, v in ((6.4, -0.35, -0.6), (6.8, -0.12, -0.6), (7.1, 0.1, -0.6), (7.4, 0.32, -0.6),
                    (7.8, -0.35, -0.68), (8.1, -0.1, -0.68)):
        tl.saccade(t, h, v)
    tl.saccade(8.6, 0.05, 0.6).saccade(9.8, 0.0, 0.0)
    tl.add_blink(1.4).add_blink(5.8).add_blink(9.9).add_blink(11.5).add_blink(11.78, 0.85)
    tl.expr("wide", [(0, 0), (8.5, 0), (8.75, 0.6), (9.4, 0.5), (9.8, 0)])
    tl.expr("brow_up", [(0, 0), (3.0, 0), (3.4, 0.25), (5.4, 0.2), (5.7, 0), (8.5, 0), (8.75, 0.7), (9.4, 0.6), (9.9, 0)])
    tl.expr("squint", [(0, 0), (10.0, 0), (10.4, 0.45), (11.3, 0.4), (11.9, 0.1)])
    tl.expr("smile", [(0, 0), (10.0, 0), (10.4, 0.55), (11.3, 0.5), (11.9, 0.2)])
    tl.expr("brow_down", [(0, 0), (6.3, 0), (6.6, 0.18), (8.2, 0.18), (8.5, 0)])
    return tl.build()


SEQUENCES = {"left": seq_side(-1), "right": seq_side(1), "motion": seq_motion}

# ================================================================ module-level API

_inst = None


def _api():
    global _inst
    if _inst is None:
        _inst = Eyes()
    return _inst


def move(h, v):
    return _api().move(h, v)


def reset():
    return _api().reset()


def blink(amount=1.0):
    return _api().blink(amount)


def render(path, camera=None):
    return _api().render(path, camera)


# ================================================================ CLI tasks


def task_clockwise(e, out_dir):
    e.clear_animation()
    e.reset()
    sc = e.scene
    sc.camera = bpy.data.objects["Cam_Main"]
    records = []
    for k in range(10):
        th = 2 * math.pi * k / 10
        e.move(math.sin(th), math.cos(th))
        st = e.state()
        rec = {"frame": k, "theta": round(th, 4), **st, "camera_matrix": e.camera_matrix()["P"],
               "render": f"renders/clockwise/frame_{k:02d}.png"}
        records.append(rec)
        e.render(os.path.join(out_dir, "renders", "clockwise", f"frame_{k:02d}.png"), samples=64)
    e.reset()
    meta = {"description": "10 clockwise gaze renders, theta_k = 2*pi*k/10, h = sin(theta), v = cos(theta), starting top",
            "units": {"rotation": "radians, XYZ Euler, world frame, relative to rest (straight ahead)",
                      "pupil_center_2D": "pixels, origin top-left, 1024x1024", "3D": "metres, Blender world"},
            "camera": e.camera_matrix(), "fixation_distance_m": e.cfg["fixation_distance"],
            "camera_distance_m": e.cfg["camera_distance"], "frames": records}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    contact_sheet([os.path.join(out_dir, "renders", "clockwise", f"frame_{k:02d}.png") for k in range(10)],
                  os.path.join(out_dir, "renders", "contact_sheet.png"), cols=5)


def task_directions(e, out_dir):
    e.clear_animation()
    res = {}
    e.reset()
    base = e.state()
    e.render(os.path.join(out_dir, "renders", "directions", "reset.png"), camera="Cam_Main", samples=48)
    for name, (h, v) in {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}.items():
        e.move(h, v)
        st = e.state()
        dx = [st[f"pupil_center_2D_{s}"][0] - base[f"pupil_center_2D_{s}"][0] for s in ("L", "R")]
        dy = [st[f"pupil_center_2D_{s}"][1] - base[f"pupil_center_2D_{s}"][1] for s in ("L", "R")]
        ok = {"left": all(d < -3 for d in dx), "right": all(d > 3 for d in dx),
              "up": all(d < -3 for d in dy), "down": all(d > 3 for d in dy)}[name]
        res[name] = {"input": [h, v], "pupil_dx_px": [round(d, 2) for d in dx], "pupil_dy_px": [round(d, 2) for d in dy],
                     "rotation_L_deg": st["eye_rotation_L_deg"], "rotation_R_deg": st["eye_rotation_R_deg"],
                     "pass": bool(ok)}
        e.render(os.path.join(out_dir, "renders", "directions", f"{name}.png"), camera="Cam_Main", samples=48)
    e.reset()
    st = e.state()
    back = max(abs(st[f"pupil_center_2D_{s}"][i] - base[f"pupil_center_2D_{s}"][i]) for s in ("L", "R") for i in (0, 1))
    res["reset"] = {"max_residual_px": round(back, 4), "pass": back < 0.5}
    res["all_pass"] = all(r["pass"] for r in res.values() if isinstance(r, dict))
    with open(os.path.join(out_dir, "renders", "directions", "direction_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("DIRECTION CHECK", json.dumps(res))
    contact_sheet([os.path.join(out_dir, "renders", "directions", f"{n}.png") for n in ("left", "up", "reset", "down", "right")],
                  os.path.join(out_dir, "renders", "directions", "directions_sheet.png"), cols=5)


IDENTITY_MIX = [
    {}, {"identity000": 0.6}, {"identity001": 0.6}, {"identity002": 0.5, "identity003": 0.3},
    {"identity003": 0.6}, {"identity004": 0.6}, {"identity001": 0.4, "identity004": 0.4}, {"identity002": 0.6},
]


def task_textures(e, out_dir):
    with open(os.path.join(ROOT, "textures", "variants.json")) as f:
        variants = json.load(f)["variants"]
    e.clear_animation()
    for i, var in enumerate(variants):
        e.set_texture(var["id"])
        e.set_identity(IDENTITY_MIX[i % len(IDENTITY_MIX)])
        e.reset()
        e.move(0.12, 0.05)
        tag = f"variant_{var['id']:02d}_{var['name']}"
        e.render(os.path.join(out_dir, "renders", "textures", f"{tag}.png"), camera="Cam_Main", samples=96)
        e.render(os.path.join(out_dir, "renders", "textures", f"{tag}_eyes.png"), camera="Cam_Eyes", samples=96)
        e.render(os.path.join(out_dir, "renders", "textures", f"{tag}_34.png"), camera="Cam_ThreeQuarter", samples=96)
    e.set_texture(4)
    e.set_identity({})
    e.reset()
    e.scene.camera = bpy.data.objects["Cam_Main"]


def contact_sheet(paths, out, cols=5, scale=2):
    imgs = []
    for p in paths:
        im = bpy.data.images.load(p, check_existing=False)
        w, h = im.size
        a = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)
        a = a[: h - h % scale, : w - w % scale].reshape(h // scale, scale, w // scale, scale, 4).mean((1, 3))
        imgs.append(a)
        bpy.data.images.remove(im)
    h, w = imgs[0].shape[:2]
    rows = int(math.ceil(len(imgs) / cols))
    sheet = np.ones((rows * h, cols * w, 4), np.float32)
    for i, a in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[(rows - 1 - r) * h:(rows - r) * h, c * w:(c + 1) * w] = a
    img = bpy.data.images.new("sheet", cols * w, rows * h, alpha=False)
    img.pixels.foreach_set(sheet.ravel())
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    choices=["left", "right", "motion", "clockwise", "directions", "textures", "all"])
    ap.add_argument("--out", default=None, help="output video path for left/right/motion")
    ap.add_argument("--texture", type=int, default=4, help="UV texture variant (1-8)")
    ap.add_argument("--samples", type=int, default=32)
    a = ap.parse_args(argv)
    e = Eyes()
    e.set_texture(a.texture)
    tasks = ["left", "right", "motion", "clockwise", "directions", "textures"] if a.input == "all" else [a.input]
    for t in tasks:
        if t in SEQUENCES:
            out = a.out if (a.out and a.input != "all") else os.path.join(ROOT, "outputs", "videos", f"{t}.mp4")
            e.render_sequence(t, out, samples=a.samples)
            print("VIDEO", out)
        elif t == "clockwise":
            task_clockwise(e, ROOT)
        elif t == "directions":
            task_directions(e, ROOT)
        elif t == "textures":
            task_textures(e, ROOT)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv:
        main(argv)
