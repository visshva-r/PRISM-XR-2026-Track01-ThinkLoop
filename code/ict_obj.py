"""Minimal, dependency-free loader for the ICT-FaceKit GNM (generic neutral mesh).

Keeps the original vertex order so every blendshape OBJ maps 1:1 onto the
neutral mesh. Coordinates are returned in ICT space (centimetres, Y-up, face
looking down +Z) unless converted with ``to_blender``.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "ict_facekit")
NEUTRAL = os.path.join(ASSETS, "generic_neutral_mesh.obj")

EYE_RANGES = {
    "L": {"outer": (21451, 22221), "inner": (22221, 23021)},
    "R": {"outer": (23021, 23791), "inner": (23791, 24591)},
}
EYE_MATERIALS = {"M_ScleraLeft", "M_IrisLeft", "M_ScleraRight", "M_IrisRight"}
DROPPED_MATERIALS = {"M_EyeBlend", "M_EyeOcclusion"}


def load_obj(path=NEUTRAL):
    verts, uvs, faces = [], [], []
    mat = None
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("vt "):
                uvs.append(line.split()[1:3])
            elif line.startswith("usemtl"):
                mat = line.split()[1]
            elif line.startswith("f "):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    a = tok.split("/")
                    vi.append(int(a[0]) - 1)
                    ti.append(int(a[1]) - 1 if len(a) > 1 and a[1] else -1)
                faces.append((mat, vi, ti))
    return np.array(verts, float), np.array(uvs, float), faces


def load_verts(path):
    with open(path) as f:
        return np.array([l.split()[1:4] for l in f if l.startswith("v ")], float)


def landmarks(V):
    with open(os.path.join(ASSETS, "vertex_indices.json")) as f:
        idx = json.load(f)["idx_to_landmark_verts"]
    return V[np.array(idx)]


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    return c, float(np.sqrt(x[3] + c @ c))


def eye_centers(V):
    out = {}
    for side, rng in EYE_RANGES.items():
        c, r = fit_sphere(V[rng["outer"][0]:rng["outer"][1]])
        out[side] = (c, r)
    return out


def to_blender(P):
    """ICT cm (x, y-up, z-forward) -> Blender metres (x, -z forward=-Y, y-up=Z)."""
    P = np.asarray(P, float)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1) * 0.01
