"""Procedural UV texture synthesis for the ICT-FaceKit GNM head.

Every texel of the GNM UV layout is rasterised back to its 3D surface position,
so skin, brows, lips, hair and irises are painted in 3D anatomical space and
written straight into the model's own UV tiles (seam-free by construction).

Outputs per variant (textures/variant_XX/):
  face_1001.png   face UV tile (u 0..1)
  head_1002.png   back-of-head / ears / neck UV tile (u 1..2)
  eye_L.png       left eyeball (sclera + iris + pupil) on the GNM eye UVs
  eye_R.png       right eyeball, independently synthesised iris
Shared maps (textures/shared/): roughness_100x.png, height_100x.png

Run:  python code/texture_synth.py            (pure numpy, no other deps)
"""
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ict_obj  # noqa: E402

ROOT = ict_obj.ROOT
TEX_DIR = os.path.join(ROOT, "textures")
SKIN_RES = 2048
EYE_RES = 1024

# ---------------------------------------------------------------- utilities


def write_png(path, img):
    """img: HxWx3 or HxW float [0,1] (sRGB). Minimal PNG encoder."""
    a = np.clip(img, 0, 1)
    a = (a * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        color_type, ch = 0, 1
        a = a[..., None]
    else:
        color_type, ch = 2, 3
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def vnoise(P, freq, seed=0):
    """3D value noise in [0,1]. P: Nx3 (any unit), freq: cycles per unit (scalar or 3-vector)."""
    Q = P * np.asarray(freq, float)
    i = np.floor(Q).astype(np.int64)
    f = (Q - i).astype(np.float32)
    f = f * f * (3 - 2 * f)
    out = np.zeros(len(P), np.float32)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out += wx * wy * wz * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm(P, freq, octaves=4, seed=0, gain=0.5):
    total, amp, norm = np.zeros(len(P), np.float32), 1.0, 0.0
    fr = np.asarray(freq, float)
    for o in range(octaves):
        total += amp * vnoise(P, fr, seed + 17 * o)
        norm += amp
        amp *= gain
        fr = fr * 2.03
    return total / norm


# ------------------------------------------------------------ rasterisation


def rasterize(V, VT, faces, mats, u0, res):
    """Return (pos HxWx3 in ICT cm, mask HxW) for faces with material in mats on UV tile u0."""
    pos = np.zeros((res, res, 3), np.float32)
    mask = np.zeros((res, res), bool)
    for mat, vi, ti in faces:
        if mat not in mats:
            continue
        for k in range(1, len(vi) - 1):
            tri = (0, k, k + 1)
            uv = VT[[ti[j] for j in tri]]
            P = V[[vi[j] for j in tri]]
            x = (uv[:, 0] - u0) * res - 0.5
            y = (1.0 - uv[:, 1]) * res - 0.5
            x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), res - 1))
            y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), res - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(d) < 1e-12:
                continue
            l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
            l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
            if not inside.any():
                continue
            yy, xx = gy[inside], gx[inside]
            pos[yy, xx] = (l0[inside, None] * P[0] + l1[inside, None] * P[1] + l2[inside, None] * P[2])
            mask[yy, xx] = True
    return pos, mask


def dilate(img, mask, iters=12):
    """Bleed valid texels outward so mip-mapping / filtering never shows seams."""
    img = img.copy()
    m = mask.astype(np.float32)
    for _ in range(iters):
        acc = np.zeros_like(img)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            acc += np.roll(np.roll(img * m[..., None] if img.ndim == 3 else img * m, dy, 0), dx, 1)
            cnt += np.roll(np.roll(m, dy, 0), dx, 1)
        grow = (m == 0) & (cnt > 0)
        if img.ndim == 3:
            img[grow] = acc[grow] / cnt[grow, None]
        else:
            img[grow] = acc[grow] / cnt[grow]
        m = np.maximum(m, grow.astype(np.float32))
    return img


# ----------------------------------------------------------- anatomy helpers


def seg_dist(P, A, B):
    """Distance from points P (Nx3) to segments A->B; returns (dist, t)."""
    AB = B - A
    t = np.clip(((P - A) @ AB) / (AB @ AB), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1), t


def polyline_dist(P, pts):
    best = np.full(len(P), 1e9, np.float32)
    best_t = np.zeros(len(P), np.float32)
    n = len(pts) - 1
    for i in range(n):
        d, t = seg_dist(P, pts[i], pts[i + 1])
        sel = d < best
        best[sel] = d[sel]
        best_t[sel] = (i + t[sel]) / n
    return best, best_t


def point_in_poly_xy(P, poly):
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i, 0], poly[i, 1]
        x2, y2 = poly[(i + 1) % n, 0], poly[(i + 1) % n, 1]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond
    return inside


def gauss(P, c, sigma):
    return np.exp(-((P - c) ** 2).sum(1) / (2 * sigma * sigma))


# --------------------------------------------------------------- variants

VARIANTS = [
    # name, skin sRGB, hair sRGB, brow sRGB, lip sRGB, freckles, stubble, rosy, under-eye, iris preset
    dict(name="fair_freckled", skin=(224, 182, 160), hair=(112, 78, 50), brow=(96, 66, 44), lip=(196, 116, 114), freckles=0.9, stubble=0.0, rosy=0.55, undereye=0.25, iris="blue"),
    dict(name="light_rosy", skin=(216, 166, 138), hair=(150, 116, 72), brow=(110, 82, 52), lip=(190, 108, 104), freckles=0.15, stubble=0.0, rosy=0.8, undereye=0.3, iris="green"),
    dict(name="medium_warm", skin=(206, 152, 118), hair=(52, 37, 28), brow=(48, 34, 26), lip=(172, 98, 88), freckles=0.0, stubble=0.25, rosy=0.45, undereye=0.35, iris="hazel"),
    dict(name="olive", skin=(190, 142, 104), hair=(28, 22, 19), brow=(32, 24, 20), lip=(160, 94, 82), freckles=0.0, stubble=0.55, rosy=0.3, undereye=0.4, iris="brown"),
    dict(name="south_asian_tan", skin=(168, 114, 82), hair=(20, 16, 14), brow=(24, 18, 15), lip=(140, 78, 70), freckles=0.0, stubble=0.35, rosy=0.25, undereye=0.55, iris="dark_brown"),
    dict(name="brown", skin=(138, 92, 66), hair=(18, 14, 12), brow=(20, 15, 13), lip=(118, 66, 58), freckles=0.0, stubble=0.65, rosy=0.2, undereye=0.5, iris="dark_brown"),
    dict(name="dark", skin=(102, 67, 48), hair=(14, 11, 10), brow=(16, 12, 10), lip=(92, 54, 48), freckles=0.0, stubble=0.3, rosy=0.15, undereye=0.45, iris="very_dark"),
    dict(name="gray_eyed_fair", skin=(218, 172, 148), hair=(70, 52, 40), brow=(64, 48, 36), lip=(182, 110, 106), freckles=0.35, stubble=0.15, rosy=0.5, undereye=0.3, iris="gray"),
]

IRIS = {
    # outer (limbal side), inner (pupillary side), collarette
    "blue": ((62, 104, 148), (150, 138, 104), (120, 150, 175)),
    "green": ((78, 112, 66), (156, 124, 58), (118, 140, 84)),
    "hazel": ((96, 92, 52), (138, 88, 38), (150, 118, 60)),
    "brown": ((92, 58, 30), (120, 76, 36), (140, 92, 44)),
    "dark_brown": ((56, 34, 20), (78, 48, 24), (92, 58, 30)),
    "very_dark": ((34, 22, 15), (50, 31, 18), (60, 38, 22)),
    "gray": ((112, 124, 132), (150, 140, 112), (150, 160, 166)),
}


def srgb(c):
    return np.array(c, np.float32) / 255.0


# ------------------------------------------------------------------ skin


class SkinFields:
    """Per-tile geometry + all noise fields shared by every skin variant."""

    def __init__(self, V, VT, faces, u0, LM):
        t = time.time()
        pos, mask = rasterize(V, VT, faces, {"M_Face", "M_BackHead"}, u0, SKIN_RES)
        pos = dilate(pos, mask, 24)
        self.mask = mask
        self.shape = mask.shape
        cover = pos.reshape(-1, 3)
        self.valid = (np.abs(cover).sum(1) > 0)
        P = cover[self.valid].astype(np.float64)
        self.P = P
        n = len(P)
        lm = LM
        self.eye_y = lm[36:48, 1].mean()
        self.brow_y = lm[17:27, 1].mean()
        self.front_z = lm[27:31, 2].max()

        # low/mid/high-frequency noise (units: cm)
        self.mottle = fbm(P, 0.35, 4, seed=1)
        self.blotch = fbm(P, 1.4, 3, seed=2)
        self.pores = vnoise(P, 22.0, seed=3)
        self.pores2 = vnoise(P, 41.0, seed=4)
        self.fine = fbm(P, 9.0, 2, seed=5)
        self.freck = vnoise(P, 7.5, seed=6) * (0.6 + 0.4 * fbm(P, 0.8, 2, seed=12))
        self.stub = vnoise(P, 30.0, seed=7)
        self.hair_n = vnoise(P * np.array([1.0, 1.0, 0.18]), 26.0, seed=8)
        self.hair_low = fbm(P, 0.9, 3, seed=9)

        # anatomical masks
        self.lips = self._lip_mask(lm)
        nose_tip = lm[30]
        self.nose = gauss(P, nose_tip + [0, -0.3, -0.4], 1.1)
        cheekL = (lm[14] + lm[54] + lm[46]) / 3 + [0.4, 0.4, 0.2]
        cheekR = (lm[2] + lm[48] + lm[41]) / 3 + [-0.4, 0.4, 0.2]
        self.cheeks = np.maximum(gauss(P, cheekL, 1.6), gauss(P, cheekR, 1.6)) * (P[:, 2] > self.front_z - 8)
        uL = lm[42:48].mean(0) + [0, -1.1, -0.3]
        uR = lm[36:42].mean(0) + [0, -1.1, -0.3]
        self.under = np.maximum(gauss(P, uL, 0.75), gauss(P, uR, 0.75))
        lidL = lm[42:48].mean(0) + [0, 0.6, 0.0]
        lidR = lm[36:42].mean(0) + [0, 0.6, 0.0]
        self.lids = np.maximum(gauss(P, lidL, 0.7), gauss(P, lidR, 0.7))
        self.tzone = np.maximum(gauss(P, lm[27] + [0, 3.2, -0.5], 2.4), self.nose)
        self.brows, self.brow_strand = self._brow_mask(lm)
        self.hair = self._hair_mask()
        self.beard = self._beard_mask(lm)
        self.ears = smoothstep(6.2, 7.2, np.abs(P[:, 0])) * (P[:, 1] < self.eye_y + 2.5) * (P[:, 1] > self.eye_y - 6.5)
        self.lip_lines = vnoise(P * np.array([1.0, 0.12, 1.0]), 14.0, seed=10)
        print(f"  tile {u0}: {n} texels, fields in {time.time() - t:.1f}s")

    def _lip_mask(self, lm):
        P = self.P
        outer = lm[48:60]
        front = P[:, 2] > lm[48:60, 2].min() - 1.2
        inside = point_in_poly_xy(P, outer) & front
        d, _ = polyline_dist(P[:, :2].copy() if False else np.c_[P[:, 0], P[:, 1], np.zeros(len(P))],
                             np.c_[np.r_[outer[:, 0], outer[:1, 0]], np.r_[outer[:, 1], outer[:1, 1]], np.zeros(13)])
        soft = np.where(inside, 1.0, 1.0 - smoothstep(0.0, 0.22, d)) * front
        return np.clip(soft, 0, 1).astype(np.float32)

    def _brow_mask(self, lm):
        P = self.P
        out = np.zeros(len(P), np.float32)
        strand = np.zeros(len(P), np.float32)
        for idx, sign in ((range(17, 22), -1), (range(22, 27), 1)):
            pts = lm[list(idx)].copy()
            pts[:, 1] += 0.15
            pts[:, 2] += 0.15
            d, t = polyline_dist(P, pts)
            # medial (inner) end is thicker: inner end = t=1 for right brow (17..21), t=0 for left
            inner = t if sign < 0 else 1 - t
            width = 0.30 + 0.32 * inner
            m = 1 - smoothstep(width * 0.55, width, d)
            m *= P[:, 2] > pts[:, 2].min() - 2.0
            # strands: medial hairs point up, lateral hairs sweep sideways
            ang = np.where(inner > 0.75, 1.2, 0.35)
            q = np.c_[P[:, 0] * np.cos(ang) - P[:, 1] * np.sin(ang), P[:, 0] * np.sin(ang) + P[:, 1] * np.cos(ang), P[:, 2]]
            s = vnoise(q * np.array([1.0, 0.16, 1.0]), 34.0, seed=11 + sign)
            out = np.maximum(out, m)
            strand = np.maximum(strand, m * s)
        return out, strand

    def _hair_mask(self):
        P = self.P
        thr = self.brow_y + 5.6 - 0.72 * (self.front_z - P[:, 2])
        jitter = (self.hair_low - 0.5) * 1.1
        m = smoothstep(-0.35, 0.45, P[:, 1] - thr + jitter)
        return (m * (1 - self.ears_mask_hint())).astype(np.float32)

    def ears_mask_hint(self):
        P = self.P
        return smoothstep(6.0, 7.0, np.abs(P[:, 0])) * (P[:, 1] < self.eye_y + 1.6) * (P[:, 1] > self.eye_y - 6.5)

    def _beard_mask(self, lm):
        P = self.P
        below_nose = smoothstep(lm[33, 1] - 0.2, lm[33, 1] - 1.0, P[:, 1])
        jaw_band = smoothstep(lm[8, 1] - 3.5, lm[8, 1] - 0.5, P[:, 1])
        cheek_line = smoothstep(lm[29, 1], lm[29, 1] - 2.8, P[:, 1])
        sides = smoothstep(7.8, 5.8, np.abs(P[:, 0]))
        front = P[:, 2] > self.front_z - 9.5
        m = np.maximum(below_nose * smoothstep(3.2, 1.5, np.abs(P[:, 0])), cheek_line * sides) * jaw_band * front
        m *= 1 - self.lips
        return np.clip(m, 0, 1).astype(np.float32)

    def to_image(self, vals, ch=3):
        img = np.zeros((self.shape[0] * self.shape[1], ch) if ch > 1 else self.shape[0] * self.shape[1], np.float32)
        img[self.valid] = vals
        return img.reshape(self.shape + ((ch,) if ch > 1 else ()))


def skin_color(F, v, seed):
    P = F.P
    base = srgb(v["skin"])
    col = np.tile(base, (len(P), 1))
    lum = 1.0 + (F.mottle - 0.5) * 0.10 + (F.blotch - 0.5) * 0.05 + (F.fine - 0.5) * 0.04
    col *= lum[:, None]
    red = np.array([0.90, 0.55, 0.52], np.float32)
    col = col * (1 - 0.16 * v["rosy"] * F.cheeks[:, None]) + red * base * (0.16 * v["rosy"] * F.cheeks[:, None]) * 1.25
    col = col * (1 - 0.12 * F.nose[:, None]) + (base * red * 1.2) * (0.12 * F.nose[:, None])
    dark = np.array([0.72, 0.62, 0.70], np.float32)
    col *= (1 - v["undereye"] * 0.22 * F.under[:, None] * (1 - dark))
    col *= (1 - 0.10 * F.lids[:, None] * (1 - np.array([0.85, 0.72, 0.72])))
    # lips with vertical lip lines
    lip = srgb(v["lip"]) * (0.92 + 0.12 * F.lip_lines[:, None])
    col = col * (1 - F.lips[:, None]) + lip * F.lips[:, None]
    # freckles
    if v["freckles"] > 0:
        fr = smoothstep(0.66 - 0.06 * v["freckles"], 0.74, F.freck) * (F.cheeks * 1.6 + F.nose * 1.4).clip(0, 1)
        col *= (1 - 0.22 * fr[:, None] * np.array([0.45, 0.70, 0.95]))
    # stubble / beard shadow
    if v["stubble"] > 0:
        dots = smoothstep(0.55, 0.85, F.stub)
        shade = F.beard * (0.35 + 0.65 * dots) * v["stubble"]
        hairc = srgb(v["hair"])
        col = col * (1 - 0.55 * shade[:, None]) + hairc * (0.55 * shade[:, None])
    # pores: tiny darkening
    pore = smoothstep(0.62, 0.92, F.pores) * 0.06 + smoothstep(0.7, 0.95, F.pores2) * 0.04
    col *= (1 - pore[:, None])
    # eyebrows (strand texture)
    bc = srgb(v["brow"])
    b = np.clip(F.brows * (0.35 + 0.65 * smoothstep(0.35, 0.75, F.brow_strand / np.maximum(F.brows, 1e-3))), 0, 1)
    col = col * (1 - 0.92 * b[:, None]) + bc * (0.92 * b[:, None])
    # scalp hair (close-cropped)
    hc = srgb(v["hair"])
    strands = 0.7 + 0.6 * F.hair_n
    hair = F.hair * (0.75 + 0.25 * smoothstep(0.3, 0.7, F.hair_n))
    col = col * (1 - hair[:, None]) + (hc * strands[:, None]) * hair[:, None]
    return np.clip(col, 0, 1)


def skin_rough_height(F):
    rough = 0.47 - 0.10 * F.tzone - 0.12 * F.lips + 0.14 * F.hair + 0.08 * F.brows + (F.fine - 0.5) * 0.06
    height = 0.5 - 0.22 * smoothstep(0.62, 0.92, F.pores) - 0.12 * smoothstep(0.7, 0.95, F.pores2) \
        + (F.fine - 0.5) * 0.18 + 0.25 * F.hair * (F.hair_n - 0.5) + 0.2 * F.brow_strand
    return np.clip(rough, 0.05, 1), np.clip(height, 0, 1)


# ------------------------------------------------------------------ eyes


class EyeFields:
    def __init__(self, V, VT, faces, side, center):
        mats = {"M_IrisLeft"} if side == "L" else {"M_IrisRight"}
        pos, mask = rasterize(V, VT, faces, mats, 0.0, EYE_RES)
        pos = dilate(pos, mask, 16)
        self.shape = mask.shape
        flat = pos.reshape(-1, 3)
        self.valid = np.abs(flat).sum(1) > 0
        D = flat[self.valid] - center
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        self.alpha = np.degrees(np.arccos(np.clip(D[:, 2], -1, 1)))  # angle from optical axis
        self.phi = np.arctan2(D[:, 1], D[:, 0])
        self.D = D
        self.side = side

    def to_image(self, vals):
        img = np.zeros((self.shape[0] * self.shape[1], 3), np.float32)
        img[self.valid] = vals
        return img.reshape(self.shape + (3,))


IRIS_DEG = 24.0
PUPIL_DEG = 8.5


def eye_color(E, preset, seed):
    a, phi, D = E.alpha, E.phi, E.D
    n = len(a)
    r = a / IRIS_DEG
    outer, inner, collar = (srgb(c) for c in IRIS[preset])
    rng = np.random.default_rng(seed)
    hue = 1.0 + rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    outer, inner, collar = outer * hue, inner * hue, collar * hue

    ring = np.c_[np.cos(phi), np.sin(phi), np.zeros(n)]
    fib = vnoise(np.c_[ring * 60.0, r * 1.8], 1.0, seed)
    fib2 = vnoise(np.c_[ring * 130.0, r * 3.5], 1.0, seed + 1)
    crypt = vnoise(np.c_[ring * 9.0, r * 6.0], 1.0, seed + 2)
    collar_r = 0.42 + 0.05 * (vnoise(np.c_[ring * 7.0, np.zeros(n)], 1.0, seed + 3) - 0.5) * 2
    t = smoothstep(collar_r - 0.05, 0.95, r)
    iris = inner[None] * (1 - t[:, None]) + outer[None] * t[:, None]
    iris *= (0.72 + 0.45 * fib[:, None]) * (0.9 + 0.2 * fib2[:, None])
    cring = np.exp(-((r - collar_r) / 0.045) ** 2)
    iris = iris * (1 - 0.55 * cring[:, None]) + collar[None] * 1.15 * (0.55 * cring[:, None])
    crypts = smoothstep(0.62, 0.82, crypt) * (r > collar_r) * (r < 0.9)
    iris *= (1 - 0.35 * crypts)[:, None]
    limbal = smoothstep(0.80, 1.0, r)
    iris *= (1 - 0.65 * limbal[:, None])

    # sclera
    scl = srgb((238, 232, 226)) * (0.97 + 0.04 * vnoise(D * 8.0, 1.0, seed + 4))[:, None]
    scl = scl * (1 - 0.06 * smoothstep(40, 80, a)[:, None]) + srgb((226, 214, 196)) * (0.06 * smoothstep(40, 80, a)[:, None])
    corner = smoothstep(0.55, 0.95, np.abs(D[:, 0])) * smoothstep(30, 70, a)
    scl = scl * (1 - 0.18 * corner[:, None]) + srgb((214, 150, 140)) * (0.18 * corner[:, None])
    vn = fbm(D * 3.0, 1.0, 4, seed + 5)
    vein = np.exp(-((vn - 0.5) / 0.012) ** 2) * smoothstep(34, 70, a)
    vn2 = fbm(D * 6.5, 1.0, 3, seed + 6)
    vein = np.maximum(vein, 0.6 * np.exp(-((vn2 - 0.5) / 0.008) ** 2) * smoothstep(40, 80, a))
    scl = scl * (1 - 0.55 * vein[:, None]) + srgb((170, 52, 48)) * (0.55 * vein[:, None])

    iris_mask = 1 - smoothstep(0.985, 1.04, r)
    col = scl * (1 - iris_mask[:, None]) + iris * iris_mask[:, None]
    pupil = 1 - smoothstep(PUPIL_DEG / IRIS_DEG - 0.03, PUPIL_DEG / IRIS_DEG + 0.02, r)
    col = col * (1 - pupil[:, None]) + srgb((6, 6, 8)) * pupil[:, None]
    return np.clip(col, 0, 1)


# ------------------------------------------------------------------ main


def main():
    t0 = time.time()
    V, VT, faces = ict_obj.load_obj()
    LM = ict_obj.landmarks(V)
    centers = ict_obj.eye_centers(V)
    print("synthesising skin fields...")
    tiles = {1001: SkinFields(V, VT, faces, 0.0, LM), 1002: SkinFields(V, VT, faces, 1.0, LM)}
    for tile, F in tiles.items():
        rough, height = skin_rough_height(F)
        write_png(os.path.join(TEX_DIR, "shared", f"roughness_{tile}.png"), dilate(F.to_image(rough, 1), F.mask, 4))
        write_png(os.path.join(TEX_DIR, "shared", f"height_{tile}.png"), dilate(F.to_image(height, 1), F.mask, 4))
    print("synthesising eye fields...")
    eyes = {s: EyeFields(V, VT, faces, s, centers[s][0]) for s in ("L", "R")}
    manifest = []
    for i, v in enumerate(VARIANTS, 1):
        d = os.path.join(TEX_DIR, f"variant_{i:02d}")
        for tile, F in tiles.items():
            img = F.to_image(skin_color(F, v, i))
            name = "face_1001.png" if tile == 1001 else "head_1002.png"
            write_png(os.path.join(d, name), img)
        for s, E in eyes.items():
            write_png(os.path.join(d, f"eye_{s}.png"), E.to_image(eye_color(E, v["iris"], 100 * i + (1 if s == "L" else 2))))
        manifest.append({"id": i, **{k: v[k] for k in ("name", "iris")}, "dir": f"textures/variant_{i:02d}"})
        print(f"  variant {i:02d} {v['name']} done ({time.time() - t0:.0f}s)")
    with open(os.path.join(TEX_DIR, "variants.json"), "w") as f:
        json.dump({"uv_layout": "ICT-FaceKit GNM (face tile u0-1, head tile u1-2, eye tiles u0-1)",
                   "variants": manifest, "iris_deg": IRIS_DEG, "pupil_deg": PUPIL_DEG}, f, indent=2)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
