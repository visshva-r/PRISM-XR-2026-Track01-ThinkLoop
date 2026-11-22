"""Stitch the submission demo video from the rendered outputs (requires ffmpeg on PATH or FFMPEG env var).

Run:  python code/make_demo.py
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
OUT = os.path.join(ROOT, "outputs", "videos", "demo_full.mp4")
FPS = 30
S = 1024


def main():
    still = []
    for p in sorted(glob.glob(os.path.join(ROOT, "renders", "textures", "variant_*_34.png"))):
        still.append((p, 1.4))
    for p in sorted(glob.glob(os.path.join(ROOT, "renders", "textures", "variant_*_eyes.png")))[:4]:
        still.append((p, 1.2))
    clips = [os.path.join(ROOT, "outputs", "videos", f"{n}.mp4") for n in ("left", "right", "motion")]
    tail = [(os.path.join(ROOT, "renders", "contact_sheet.png"), 4.0),
            (os.path.join(ROOT, "renders", "directions", "directions_sheet.png"), 4.0)]

    args, filters, labels = [FFMPEG, "-y", "-v", "error"], [], []
    idx = 0
    norm = f"scale={S}:{S}:force_original_aspect_ratio=decrease,pad={S}:{S}:(ow-iw)/2:(oh-ih)/2:color=0x0d0f13,setsar=1,fps={FPS},format=yuv420p"
    for p, d in still:
        args += ["-loop", "1", "-t", str(d), "-i", p]
        filters.append(f"[{idx}:v]{norm}[v{idx}]")
        labels.append(f"[v{idx}]")
        idx += 1
    for p in clips:
        args += ["-i", p]
        filters.append(f"[{idx}:v]{norm}[v{idx}]")
        labels.append(f"[v{idx}]")
        idx += 1
    for p, d in tail:
        args += ["-loop", "1", "-t", str(d), "-i", p]
        filters.append(f"[{idx}:v]{norm}[v{idx}]")
        labels.append(f"[v{idx}]")
        idx += 1
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]")
    args += ["-filter_complex", ";".join(filters), "-map", "[out]", "-c:v", "libx264", "-crf", "18",
             "-preset", "medium", "-pix_fmt", "yuv420p", OUT]
    subprocess.run(args, check=True)
    print("DEMO", OUT)


if __name__ == "__main__":
    sys.exit(main())
