from pathlib import Path
import os

def count_images_in_dir(dirpath):
    count = 0
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
    return count

paths = [
    r"C:\Users\Acer\Downloads\archive (5)",
    r"C:\Users\Acer\Downloads\archive (4)"
]

for p in paths:
    pth = Path(p)
    if not pth.exists():
        print(f"{p} : NOT FOUND")
        continue
    print(f"\nDataset: {pth}")
    subdirs = sorted([x for x in pth.iterdir() if x.is_dir()])
    if not subdirs:
        print("  (no class subfolders found)\n")
        continue
    total = 0
    for d in subdirs:
        c = count_images_in_dir(str(d))
        total += c
        print(f"  {d.name}: {c} images")
    print(f"  TOTAL images in {pth.name}: {total}\n")
