"""
Delete existing trained models and checkpoints to prepare for a fresh training run.
This script will remove files under `models/checkpoints` and `models/saved`, and will
optionally remove the auto-generated landmarks in `data/landmarks` and processed data.
Use with caution: this permanently deletes files.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / 'models' / 'checkpoints'
SAVED = ROOT / 'models' / 'saved'
LANDMARKS = ROOT / 'data' / 'landmarks'
PROCESSED = ROOT / 'data' / 'processed'

print('Preparing to delete existing models and caches...')
print(f'Checkpoints: {CHECKPOINTS}')
print(f'Saved models: {SAVED}')
print(f'Landmarks: {LANDMARKS}')
print(f'Processed data: {PROCESSED}')

confirm = input('Type YES to confirm permanent deletion: ').strip()
if confirm != 'YES':
    print('Aborted by user.')
    raise SystemExit(0)

for p in [CHECKPOINTS, SAVED]:
    if p.exists():
        print(f'Deleting {p} ...')
        try:
            shutil.rmtree(p)
            print('Deleted')
        except Exception as e:
            print(f'Error deleting {p}: {e}')
    else:
        print(f'Not found: {p}')

# Optionally delete landmarks/processed to start fresh
for p in [LANDMARKS, PROCESSED]:
    if p.exists():
        print(f'Deleting {p} ...')
        try:
            shutil.rmtree(p)
            print('Deleted')
        except Exception as e:
            print(f'Error deleting {p}: {e}')
    else:
        print(f'Not found: {p}')

print('\nCleanup complete.')
