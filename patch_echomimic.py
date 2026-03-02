"""Build-time patch: adjust EchoMimic's face mask to start at face-box top.

The default mask starts ABOVE the MTCNN bounding box (rb - r_pad), which
includes forehead/hair area and can cause one-eye-winking artefacts.

We change the top edge to start exactly at `rb` (top of face bbox).
This includes the full face (eyes, brows, nose, mouth) in the animation
zone while excluding the forehead padding that caused winking.
"""
import sys

path = "/app/EchoMimic/infer_audio2vid.py"

with open(path) as f:
    src = f.read()

old = "face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255"
new = "face_mask[rb : re + r_pad, cb - c_pad : ce + c_pad] = 255"
# rb = exact top of face bbox (includes eyes + brows)
# rb - r_pad = original (extends above face → caused winking)
# rb + 0.2*(re-rb) = previous patch (cut off eyes → no blinking/expression)

if old not in src:
    print(f"ERROR: patch target not found in {path}. Check EchoMimic version.", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)

print("✅ Patched: face mask starts at face-box top (full face animated).")
