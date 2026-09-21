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

def generate_for_system(data_path, out_dir, system_name, expected_letters):
    print(f"\nGenerating skeleton images for {system_name} ({len(expected_letters)} letters)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = data["landmarks"]
    y = data["labels"]
    class_names = data.get("class_names", [chr(65 + i) for i in range(len(set(y)))])

    for letter_idx, letter_char in enumerate(expected_letters):
        # Match by class_names index if present, else letter_idx
        if letter_char in class_names:
            cls_idx = class_names.index(letter_char)
        else:
            cls_idx = letter_idx

        indices = [i for i, l in enumerate(y) if l == cls_idx]
        if not indices:
            print(f"  Warning: No samples for letter {letter_char}")
            continue

        # For signs with two hands, prioritize samples that contain both hands detected
        both_indices = [i for i in indices if np.count_nonzero(X[i][:21]) > 0 and len(X[i]) >= 42 and np.count_nonzero(X[i][21:42]) > 0]
        pool = both_indices if len(both_indices) >= 5 else indices

        # Choose the median/central sample across the pool to ensure good representation
        samples = [np.array(X[i]).reshape(-1, 3) for i in pool[:150]]
        flat_samples = np.array([s.flatten() for s in samples])
        mean_sample = flat_samples.mean(axis=0)
        dists = np.linalg.norm(flat_samples - mean_sample, axis=1)
        best_sample = samples[np.argmin(dists)]

        left_pts = best_sample[:21]
        right_pts = best_sample[21:42] if len(best_sample) >= 42 else None

        # For ASL or single-hand samples where right hand had the data instead of left:
        if left_pts is not None and np.all(left_pts == 0) and right_pts is not None and not np.all(right_pts == 0):
            left_pts, right_pts = right_pts, None

        out_file = out_dir / f"{letter_char}.png"
        render_skeleton_image(left_pts, right_pts, out_file)
        print(f"  Saved {out_file.name} (from {len(indices)} samples)")

    print(f"Finished {system_name} skeleton guides in {out_dir}")

def main():
    isl_data = ROOT / "data" / "landmarks" / "landmarks_letter_hands_only_mlp.pkl"
    isl_out = ROOT / "mobile" / "app" / "pamphlet" / "isl-skeleton"
    asl_data = ROOT / "data" / "landmarks" / "landmarks_asl_hands_only_mlp.pkl"
    asl_out = ROOT / "mobile" / "app" / "pamphlet" / "asl-skeleton"

    if isl_data.exists():
        generate_for_system(isl_data, isl_out, "ISL", [chr(65 + i) for i in range(26)])
    if asl_data.exists():
        asl_letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in ('J', 'Z')]
        generate_for_system(asl_data, asl_out, "ASL", asl_letters)

if __name__ == "__main__":
    main()

