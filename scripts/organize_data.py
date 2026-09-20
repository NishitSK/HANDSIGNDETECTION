"""
Organize datasets from the two archive folders into data/ISL_DATASETS_2/<class> structure.
This script copies class subfolders from the archive paths to the project's `data/ISL_DATASETS_2` folder.
It skips files that already exist to avoid re-copying.
"""
import shutil
from pathlib import Path

ARCHIVES = [
    Path(r"C:/Users/Acer/Downloads/archive (5)"),
    Path(r"C:/Users/Acer/Downloads/archive (4)")
]
DEST_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'ISL_DATASETS_2'

DEST_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Destination root: {DEST_ROOT}")

for src in ARCHIVES:
    if not src.exists():
        print(f"Source not found: {src}")
        continue

    print(f"Processing archive: {src}")
    # For each top-level directory in src, treat as class folder
    for class_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        dest_dir = DEST_ROOT / class_dir.name
        if not dest_dir.exists():
            print(f"Copying class folder: {class_dir.name}")
            try:
                shutil.copytree(class_dir, dest_dir)
            except Exception as e:
                print(f"Error copying {class_dir}: {e}")
        else:
            # Merge: copy files inside class_dir into dest_dir
            print(f"Merging into existing class folder: {class_dir.name}")
            for src_file in class_dir.rglob('*'):
                if src_file.is_file():
                    rel = src_file.relative_to(class_dir)
                    target_file = dest_dir / rel
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        try:
                            shutil.copy2(src_file, target_file)
                        except Exception as e:
                            print(f"Failed to copy {src_file}: {e}")
print("\nData organization complete.")
