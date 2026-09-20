import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "landmarks" / "landmarks_letter_hands_only_mlp.pkl"
OUT_DIR = ROOT / "mobile" / "app" / "pamphlet" / "isl-skeleton"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def render_skeleton_image(points_left, points_right, out_path, size=512):
    # Collect all non-zero points
    all_pts = []
    if points_left is not None and not np.all(points_left == 0):
        all_pts.append(points_left)
    if points_right is not None and not np.all(points_right == 0):
        all_pts.append(points_right)

    if not all_pts:
        return

    combined = np.vstack(all_pts)
    min_x, max_x = combined[:, 0].min(), combined[:, 0].max()
    min_y, max_y = combined[:, 1].min(), combined[:, 1].max()

    # Calculate scale to fit in size with 18% padding
    pad = 0.18 * size
    target_w = size - 2 * pad
    target_h = size - 2 * pad

    span_x = max(max_x - min_x, 1e-4)
    span_y = max(max_y - min_y, 1e-4)
    scale = min(target_w / span_x, target_h / span_y)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    def to_img(pt):
        x = (pt[0] - cx) * scale + (size / 2)
        y = (pt[1] - cy) * scale + (size / 2)
        return (float(x), float(y))

    # Transparent RGBA canvas
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    hands_to_draw = []
    if points_left is not None and not np.all(points_left == 0):
        hands_to_draw.append((points_left, (56, 189, 248, 230), (255, 255, 255, 255))) # Cyan
    if points_right is not None and not np.all(points_right == 0):
        hands_to_draw.append((points_right, (255, 178, 63, 230), (255, 255, 255, 255))) # Marigold

    # If only 1 hand is present, use golden marigold theme
    if len(hands_to_draw) == 1:
        hands_to_draw = [(hands_to_draw[0][0], (255, 178, 63, 240), (255, 255, 255, 255))]

    for points, bone_color, joint_color in hands_to_draw:
        coords = [to_img(p) for p in points]

        # Draw bones (glow + core)
        for s, e in CONNECTIONS:
            p1, p2 = coords[s], coords[e]
            # Outer subtle glow line
            draw.line([p1, p2], fill=(bone_color[0], bone_color[1], bone_color[2], 80), width=9)
            # Core bone line
            draw.line([p1, p2], fill=bone_color, width=4)

        # Draw joints
        for i, pt in enumerate(coords):
            r = 7 if i in (0, 4, 8, 12, 16, 20) else 5
            # Glow
            draw.ellipse([pt[0] - r - 2, pt[1] - r - 2, pt[0] + r + 2, pt[1] + r + 2],
                         fill=(bone_color[0], bone_color[1], bone_color[2], 100))
            # Core joint
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=joint_color)
            # Outline
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], outline=(11, 22, 51, 220), width=2)

    img.save(out_path, "PNG")

def main():
    print(f"Loading landmarks from {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X = data["landmarks"]
    y = data["labels"]

    print("Generating skeleton images for ISL A-Z...")
    for letter_idx in range(26):
        letter_char = chr(65 + letter_idx)
        indices = [i for i, l in enumerate(y) if l == letter_idx]
        if not indices:
            print(f"Warning: No samples for letter {letter_char}")
            continue

        # Choose the median/central sample across the class to ensure good representation
        samples = [np.array(X[i]).reshape(-1, 3) for i in indices[:150]]
        # Compute mean coordinates for non-zero points
        # To pick a clean real sample: find the one closest to the class mean
        flat_samples = np.array([s.flatten() for s in samples])
        mean_sample = flat_samples.mean(axis=0)
        dists = np.linalg.norm(flat_samples - mean_sample, axis=1)
        best_sample = samples[np.argmin(dists)]

        left_pts = best_sample[:21]
        right_pts = best_sample[21:42]

        out_file = OUT_DIR / f"{letter_char}.png"
        render_skeleton_image(left_pts, right_pts, out_file)
        print(f"  Saved {out_file.name} (from {len(indices)} samples)")

    print(f"All 26 skeleton guides generated in {OUT_DIR}")

if __name__ == "__main__":
    main()
