# Track 01 - Synthetic Eye Movement & Virtual Camera Rendering

**Team:** SRM Institute of Science and Technology_Think Loop
**Event:** Samsung PRISM XR Hackathon 2026 (XR Perception Group)

A fully scripted Blender pipeline that turns the open **GNM (ICT-FaceKit generic neutral mesh)** head into a
photoreal, articulated synthetic-gaze generator. Both eyeballs rotate spherically about their fitted centres,
converge on a fixation target 35 cm ahead, and drive eyelids, brows and cheeks through the GNM blendshapes, so
every frame comes with exact ground-truth gaze, 3D pupil positions, 2D pupil centres and the full camera matrix.

| Deliverable (evaluation deck) | Where |
|---|---|
| 5-10 UV texture maps (face + eyes) for the GNM UV layout | `textures/variant_01..08/` (8 sets) |
| Python script with inputs **Left / Right / Generate Motion** that renders frames and builds a video | `wrapper/eyes.py --input left|right|motion` |
| Rendered video for each of the 3 inputs (each <= 15 s) | `outputs/videos/left.mp4` (7 s), `right.mp4` (7 s), `motion.mp4` (12 s) |
| 5-10 rendered images with different textures | `renders/textures/` (8 variants x front / eye close-up / three-quarter) |
| `.blend` that loads the head and animates an expression on the Timeline | `scene.blend` (frames 1-360 = "Generate Motion") |
| Slide-spec: `eyes.move(h,v)` / `eyes.reset()` / `eyes.render()` | `wrapper/eyes.py` |
| 10 clockwise renders, theta_k = 2*pi*k/10 | `renders/clockwise/frame_00..09.png`, `renders/contact_sheet.png` |
| Per-frame ground-truth JSON | `metadata.json` |
| 5-direction pass/fail check (L / R / U / D / reset) | `renders/directions/direction_check.json` (all pass) |

## Repository layout

```
scene.blend                 textured GNM head, eye bones, materials, cameras, lights, animated Timeline
wrapper/eyes.py             control wrapper: move / reset / blink / expression / set_texture / set_identity / render
code/ict_obj.py             order-preserving OBJ loader for the GNM + eye-centre sphere fitting
code/texture_synth.py       procedural UV texture synthesis (skin, brows, lips, hair, stubble, freckles, irises)
code/build_scene.py         builds scene.blend from the GNM assets + textures
assets/ict_facekit/         GNM neutral mesh + the blendshapes used (MIT, USC ICT)
textures/                   8 texture variants + shared roughness / height maps
renders/                    clockwise, directions and texture-variant renders
outputs/videos/             left.mp4, right.mp4, motion.mp4
metadata.json               ground truth for the 10 clockwise renders
run.bat / run.sh            one-command reproduction
```

## Setup

1. Install **Blender 4.2+** (developed and tested on Blender 5.2.4 LTS; EEVEE renderer). A portable zip build works.
2. Python 3.9+ with `numpy` is only needed to regenerate the textures (`pip install -r requirements.txt`).
3. Point the scripts at Blender:

```bat
set BLENDER_EXE=C:\path\to\blender.exe          (Windows)
export BLENDER_EXE=/path/to/blender             (Linux / macOS)
```

## Run

```bat
:: everything (textures -> scene.blend -> videos, clockwise renders, metadata, direction check, texture renders)
run.bat all

:: the three required inputs (each renders frames and encodes an H.264 MP4)
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input left
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input right
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input motion

:: optional flags
--out my_video.mp4        custom output path
--texture 5               pick UV texture variant 1-8
--samples 64              EEVEE samples per frame

:: slide-spec outputs
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input clockwise    :: renders/clockwise + metadata.json
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input directions   :: L/R/U/D/reset pass-fail
"%BLENDER_EXE%" -b scene.blend --python wrapper/eyes.py -- --input textures     :: one render set per variant
```

Python API inside Blender (Scripting tab, with `wrapper/` on `sys.path`):

```python
import eyes
eyes.move(-1, 0)        # look left (image left)
eyes.move(0.5, 0.8)     # up-right
eyes.blink(1.0)         # close lids
eyes.reset()            # centre, lids open
eyes.render("out.png")  # 1024x1024 PNG from the locked 40 cm camera

e = eyes.Eyes()
e.expression(wide=0.6, brow_up=0.7)         # surprise
e.expression(squint=0.4, smile=0.5)         # Duchenne smile
e.set_texture(6); e.set_identity({"identity001": 0.6})
print(e.state())                            # rotations, pupil 2D/3D, vergence
```

Open `scene.blend` and press **Space** to play the "Generate Motion" expression on the Timeline.

## How it works

**Head model and rig.** The GNM neutral mesh is imported with its original vertex order, so all ICT-FaceKit
blendshapes apply 1:1 as shape keys (24 expression keys for blinks, gaze-lid coupling, squint, wide, brows, cheek
and smile, plus 5 identity keys). The two eyeballs are split into their own objects. Their rotation centres are found
by least-squares sphere fitting, and each is skinned to an `eye.L` / `eye.R` bone. Both bones use a Damped-Track
constraint to a single `GazeTarget` placed **35 cm in front of the eye midpoint**, so the eyes rotate as true 3D
spheres about their centres and **converge** on the fixation point (vergence is reported per frame).

**Inputs and calibration.** `move(h, v)` with `h, v` in [-1, 1] slides the fixation point over the 35 cm plane
(+-20 cm horizontally, +-15 cm vertically, about +-30 deg / +-23 deg of eye rotation). `+h` is image-right and `+v`
is up, so "Left" means toward the left of the rendered image. The locked render camera sits **40 cm in front of the eye
midpoint**, 1024x1024 PNG, under constant studio lighting. Clockwise sampling uses `h = sin(theta_k)`,
`v = cos(theta_k)`, `theta_k = 2*pi*k/10`, starting at the top.

**Natural motion ("human-like" brownie points).** The motion synthesiser builds each sequence from measured human
oculomotor primitives:
- **Saccades** follow the main sequence (duration = 21 ms + 2.2 ms/deg) with a minimum-jerk profile and a small overshoot and glissade on large saccades.
- **Smooth pursuit** drives the circular eye roll.
- **Fixational drift and tremor** keep the eyes from ever being perfectly still.
- **Blinks** use a fast 85 ms close, a 40 ms hold and a slower 170 ms open, and often fire with large saccades. **Bell's phenomenon** makes the eyes dip slightly during each blink.
- **Lid-gaze coupling**: the upper lid follows vertical gaze and the lids deform toward the gaze side. The brows lift when looking up.
- **Expressions** include surprise (wide eyes and raised brows), a Duchenne smile (squint, cheek raise and smile), and a frown while reading.

`motion.mp4` shows a scan left and right, an up-left glance, a full eye roll, a reading pattern (rightward saccades
plus a return sweep), surprise, a smile, and a double blink.

**UV textures (texture photorealism).** `texture_synth.py` rasterises the GNM UV layout back to 3D surface positions
(face tile u 0-1, head/ears/neck tile u 1-2, eyeball tiles). It then paints in anatomical 3D space using the 68 GNM
facial landmarks, which keeps every map seam-free by construction:
- **Skin:** multi-octave mottling, pores, cheek and nose redness, under-eye darkening, and eyelid tint.
- **Hair and face details:** lips with vertical lip lines, strand-textured eyebrows (medial hairs point up, lateral hairs sweep sideways), a close-cropped scalp hairline, beard stubble, and freckles.
- **Shared maps:** roughness (T-zone and lips shinier, hair matte) and a pore height map for bump.
- **Eyes:** a separate, independently seeded iris for each eye, with radial stroma fibres, collarette, crypts, limbal ring, pupil, sclera tint, canthus pinking, and fine veins.

The eye uses a two-layer shading model: a textured inner eyeball, plus a clear Fresnel cornea and tear film for the wet
highlight. Skin uses Principled BSDF with subsurface scattering (Jensen skin mean-free-paths), roughness and bump maps.
There are 8 variants spanning a range of skin tones and iris colours (blue, green, hazel, brown, dark brown, very
dark, grey), each also rendered with a different GNM identity blend.

**Ground truth (`metadata.json`).** For each frame it records `theta`, the `input {h, v}`, per-eye world-frame XYZ
Euler rotations (radians, plus degrees), gaze direction, 3D eye centre and pupil, the **2D pupil centre in pixels**,
the fixation target, vergence, and the 3x4 OpenCV projection matrix. The full intrinsics and extrinsics are stored
once at the top.

## Verification

`renders/directions/direction_check.json` contains the automated check. It measures pupil-centre displacement in the
rendered image against the reset pose. Left, Right, Up and Down each move both pupils more than 20 px in the correct
direction, and Reset returns with a residual of 0 px. All 5 of 5 pass.

## Credits and licences

- GNM head and blendshapes: [ICT-FaceKit](https://github.com/ICT-VGL/ICT-FaceKit), USC Institute for Creative Technologies, MIT licence (`assets/ict_facekit/LICENSE_ICT_FaceKit.txt`).
- All textures are procedurally generated by this project (no scanned or third-party imagery).
- Lighting environment: Blender's bundled studio-light HDRI.
