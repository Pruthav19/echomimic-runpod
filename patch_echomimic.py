"""
Build-time patch: shift EchoMimic's face mask to start from nose level.

The default mask starts at the top of the MTCNN bounding box (eye level),
causing a one-eye-winking artefact. This patch shifts the top edge 33%
down into the bbox (nose bridge) so only the mouth/chin region is animated.
"""
import sys

path = "/app/EchoMimic/infer_audio2vid.py"

with open(path) as f:
    src = f.read()

old = "face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255"
new = "face_mask[rb + int((re - rb) * 0.33) : re + r_pad, cb - c_pad : ce + c_pad] = 255"

if old not in src:
    print(f"ERROR: patch target not found in {path}. Check EchoMimic version.", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)

print("✅ Patched: face mask now starts from nose bridge level.")
