"""Fill the organiser submission template with this project's content and renders.

Run:  python code/make_deck.py <template.pptx> <repo_url>
"""
import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Emu, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM = "Think Loop"
COLLEGE = "SRM Institute of Science and Technology"
OUT = os.path.join(ROOT, "docs", f"{COLLEGE}_{TEAM}.pptx")
TOP = 900000
W = 9144000
H = 5143500
WHITE = RGBColor(0xF4, 0xF7, 0xFB)
INK = RGBColor(0x1F, 0x23, 0x2B)
ACCENT = RGBColor(0x14, 0x28, 0xA0)


def text(slide, x, y, w, h, lines, size=14, bold_first=False, color=WHITE):
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet = line.startswith("- ")
        run = p.add_run()
        run.text = ("\u2022 " + line[2:]) if bullet else line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold_first and i == 0
        p.space_after = Pt(4)
    return tb


def image(slide, path, x, y, w=None, h=None):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, Emu(x), Emu(y), Emu(w) if w else None, Emu(h) if h else None)
    return None


def box(slide, x, y, w, h, label, fill=RGBColor(0xE8, 0xED, 0xFA)):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = ACCENT
    tf = s.text_frame
    tf.word_wrap = True
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(9 if i else 10)
        r.font.bold = i == 0
        r.font.color.rgb = INK
    return s


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(x1), Emu(y1), Emu(x2), Emu(y2))
    c.line.color.rgb = ACCENT
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(tail)


def set_text(shape, value, size=None):
    tf = shape.text_frame
    first = tf.paragraphs[0]
    runs = first.runs
    if runs:
        runs[0].text = value
        for r in runs[1:]:
            r.text = ""
        if size:
            runs[0].font.size = Pt(size)
    else:
        first.add_run().text = value
    for p in tf.paragraphs[1:]:
        for r in p.runs:
            r.text = ""


def main(template, repo):
    prs = Presentation(template)
    s = prs.slides
    R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
    checks = {}
    if os.path.exists(R("renders", "directions", "direction_check.json")):
        with open(R("renders", "directions", "direction_check.json")) as f:
            checks = json.load(f)

    # 1 - title
    for sh in s[0].shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith("Track No"):
            set_text(sh, "Track 01")
        elif t.startswith("XR Hackathon") or t.startswith("Photoreal"):
            set_text(sh, "Synthetic Gaze\non the GNM Head", 32)
        elif t.startswith("Team Name"):
            set_text(sh, f"{TEAM}\n{COLLEGE}", 16)
        elif t.startswith("Members"):
            set_text(sh, "Members: Visshva R (solo)", 12)
    text(s[0], 502920, 3600000, 7800000, 900000,
         ["Track 01  |  Synthetic Eye Movement & Virtual Camera Rendering", repo], 16)

    # 2 - problem
    text(s[1], 365760, TOP, 5300000, 4000000, [
        "XR gaze estimators need huge, exactly-labelled eye datasets.",
        "- Real capture across extreme gaze angles is slow, costly and privacy-restricted.",
        "- Simple 2D pupil sliding looks fake: real eyes rotate as spheres about their centres, converge on a target, and drag the eyelids and brows with them.",
        "- Synthetic data only helps if it is photoreal (skin, iris, wet cornea) and seam-free on a standard topology (GNM / FLAME UVs), so textures transfer directly.",
        "- Every frame needs ground truth: rotations, 2D/3D pupil centres and the camera matrix.",
        "Goal: a scriptable Blender generator with Left / Right / Generate-Motion inputs that produces natural, human-like eye and eyelid motion with photoreal textures.",
    ], 12)
    image(s[1], R("renders", "textures", "variant_04_olive_eyes.png"), 5800000, TOP + 100000, h=3900000)

    # 3 - solution
    text(s[2], 365760, TOP, 5000000, 4100000, [
        "GNM (ICT-FaceKit) head, fully driven from Python:",
        "- Eye bones at sphere-fitted eyeball centres. Both Damped-Track one fixation target 35 cm ahead, giving true spherical rotation plus vergence.",
        "- 24 GNM blendshapes for blink, lid-gaze coupling, squint, wide, brows, cheeks and smile, plus 5 identity shapes.",
        "- 8 procedurally synthesised UV texture sets (skin + separate left/right iris), painted in 3D anatomical space, so they are seam-free on GNM UVs.",
        "- Human oculomotor model: main-sequence saccades, pursuit eye-roll, drift, blinks, Bell's phenomenon.",
        "- eyes.move(h,v) / reset() / blink() / expression() / render(). The CLI renders the Left / Right / Motion videos plus metadata.",
    ], 12)
    image(s[2], R("renders", "textures", "variant_03_medium_warm.png"), 5500000, TOP, h=4000000)

    # 4 - architecture
    y0 = TOP + 150000
    bw, bh = 1950000, 900000
    xs = [250000, 2450000, 4650000, 6850000]
    box(s[3], xs[0], y0, bw, bh, "GNM assets\nICT-FaceKit neutral mesh\n+ 29 blendshape OBJs")
    box(s[3], xs[1], y0, bw, bh, "texture_synth.py\nUV raster -> 3D positions\nlandmark-driven painting")
    box(s[3], xs[2], y0, bw, bh, "8 UV texture sets\nface 1001 + head 1002\neye_L / eye_R + PBR maps")
    box(s[3], xs[3], y0, bw - 150000, bh, "build_scene.py\nshape keys, eye rig,\nmaterials, cameras, lights")
    y1 = y0 + bh + 500000
    box(s[3], xs[3], y1, bw - 150000, bh, "scene.blend\nanimated Timeline\n(Generate Motion)")
    box(s[3], xs[2], y1, bw, bh, "wrapper/eyes.py\nmove / reset / blink /\nexpression / render")
    box(s[3], xs[1], y1, bw, bh, "Motion synthesiser\nsaccades, pursuit, drift,\nblinks, expressions")
    box(s[3], xs[0], y1, bw, bh, "Outputs\nleft / right / motion .mp4\nclockwise PNG + metadata")
    for a, b in ((0, 1), (1, 2), (2, 3)):
        arrow(s[3], xs[a] + bw, y0 + bh // 2, xs[b], y0 + bh // 2)
    arrow(s[3], xs[3] + (bw - 150000) // 2, y0 + bh, xs[3] + (bw - 150000) // 2, y1)
    for a, b in ((3, 2), (2, 1), (1, 0)):
        arrow(s[3], xs[a], y1 + bh // 2, xs[b] + bw, y1 + bh // 2)
    text(s[3], 250000, y1 + bh + 150000, 8600000, 700000, [
        "Gaze: target = eye_mid + (0.20 h, 0.35 fwd, 0.15 v) m. Each eye bone Damped-Tracks it, so the eyeballs rotate about their own "
        "centres and converge. The camera is locked 40 cm ahead (1024x1024). Ground truth is read back from the evaluated rig "
        "and projected with the exact camera intrinsics and extrinsics.",
    ], 10)

    # 5 - final output
    for sh in s[4].shapes:
        if sh.has_text_frame and "Track Final" in sh.text_frame.text:
            set_text(sh, "TRACK FINAL OUTPUT")
    text(s[4], 365760, TOP - 60000, 8400000, 300000,
         ["10 clockwise renders (theta_k = 2*pi*k/10), 8 UV texture variants, 3 videos (<= 15 s), metadata.json"], 11)
    image(s[4], R("renders", "contact_sheet.png"), 365760, TOP + 280000, w=5000000)
    vs = ["variant_01_fair_freckled", "variant_02_light_rosy", "variant_05_south_asian_tan", "variant_06_brown",
          "variant_07_dark", "variant_08_gray_eyed_fair"]
    for i, v in enumerate(vs):
        r, c = divmod(i, 3)
        image(s[4], R("renders", "textures", f"{v}.png"), 5500000 + c * 1080000, TOP + 280000 + r * 1080000, w=1060000)
    image(s[4], R("renders", "textures", "variant_02_light_rosy_eyes.png"), 365760 + 5000000 + 140000, TOP + 2500000, w=1550000)
    image(s[4], R("renders", "textures", "variant_05_south_asian_tan_34.png"), 365760 + 5000000 + 1750000, TOP + 2500000, w=1550000)
    if checks:
        rows = [f"{k.capitalize()}: {'PASS' if v.get('pass') else 'FAIL'}" for k, v in checks.items() if isinstance(v, dict)]
        text(s[4], 365760, TOP + 2400000, 5000000, 1400000,
             ["Automated 5-direction check (pupil shift in pixels vs reset):", "  ".join(rows),
              "Videos: outputs/videos/left.mp4 (7 s), right.mp4 (7 s), motion.mp4 (12 s)",
              "Timeline: scene.blend plays the Generate Motion expression (frames 1-360)"], 11)

    # 6 - differentiation
    text(s[5], 365760, TOP, 8400000, 4100000, [
        "- True 3D oculomotor rig: spherical eyeball rotation about fitted centres, plus binocular vergence on a 35 cm fixation target. No 2D pupil translation.",
        "- Human motion statistics, not keyframe guesses: saccade main sequence (21 ms + 2.2 ms/deg) with minimum-jerk profile and overshoot, smooth-pursuit eye roll, fixational drift, blink dynamics (85 / 40 / 170 ms), Bell's phenomenon, lid-gaze coupling.",
        "- Seam-free by construction: textures are painted in 3D on the GNM's own UV layout (face + head tiles), with dilation padding and no projection seams.",
        "- Independent left/right iris synthesis (stroma fibres, collarette, crypts, limbal ring) under a two-layer eye shader (textured globe plus Fresnel cornea and tear film).",
        "- Dataset-ready ground truth per frame: Euler rotations, gaze vectors, 2D/3D pupils, vergence, OpenCV K / R / t / P.",
        "- Fully reproducible: one command rebuilds textures, scene, videos and metadata. Pure Python + numpy + Blender, MIT-licensed head.",
    ], 12)

    # 7 - demo & setup
    text(s[6], 365760, TOP, 8400000, 4100000, [
        f"Repository: {repo}",
        "Setup: Blender 4.2+ (tested 5.2.4 LTS) and Python 3.9+ with numpy;  set BLENDER_EXE=<path to blender.exe>",
        "Rebuild everything:  run.bat all      (or ./run.sh all)",
        "Required inputs:",
        "- blender -b scene.blend --python wrapper/eyes.py -- --input left",
        "- blender -b scene.blend --python wrapper/eyes.py -- --input right",
        "- blender -b scene.blend --python wrapper/eyes.py -- --input motion",
        "Spec outputs:  --input clockwise  (10 renders + metadata.json)   --input directions   --input textures",
        "Interactive: open scene.blend and press Space to play the animated expression on the Timeline.",
        "Demo videos: outputs/videos/left.mp4, right.mp4, motion.mp4 (H.264, 1024x1024, 30 fps)",
    ], 12)

    # 8 - brownie
    text(s[7], 365760, TOP, 4300000, 4100000, [
        "Brownie features:",
        "- Expressions: surprise (eyeWide + brow raise), Duchenne smile (squint + cheek + smile), focused frown while reading.",
        "- Reading pattern: rightward micro-saccades plus a return sweep with lowered lids.",
        "- Identity control: 5 GNM identity shapes. Eye pivots follow the reshaped sockets.",
        "- Strand-textured brows, close-cropped hairline, stubble, freckles, under-eye tone.",
        "- Fine eyelash shader with sparser lower lashes.",
        "- Three render cameras: locked 40 cm front, eye close-up, three-quarter (shows seam-free UVs).",
    ], 12)
    image(s[7], R("renders", "textures", "variant_01_fair_freckled_eyes.png"), 4800000, TOP, w=2000000)
    image(s[7], R("renders", "textures", "variant_06_brown_eyes.png"), 6900000, TOP, w=2000000)
    image(s[7], R("renders", "textures", "variant_04_olive_34.png"), 4800000, TOP + 2050000, w=2000000)
    image(s[7], R("renders", "textures", "variant_07_dark_34.png"), 6900000, TOP + 2050000, w=2000000)

    # 9 - thank you
    for sh in s[8].shapes:
        if sh.has_text_frame and sh.name == "Text 25":
            set_text(sh, f"{TEAM} | {COLLEGE} | {repo}", 10)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print("DECK", OUT)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
