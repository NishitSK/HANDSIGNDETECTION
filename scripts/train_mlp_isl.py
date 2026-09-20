"""
Retrain ISL letters as a single-frame MLP, hands-only (no face).

Reuses the already-extracted raw landmarks (no MediaPipe re-run needed) by
slicing off the face points before they're normalized/flattened — see
mobile/tests/export_reference.py for why the first 42 of 62 landmarks are
exactly "both hands" regardless of downstream feature layout.

Writes to results/ISL_MLP/ (isolated from the production model at
models/saved/isl_model.h5, which this does not touch).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import ISLTrainer

SRC = 'data/landmarks/landmarks_letter_both_hands_face.pkl'
DST = 'data/landmarks/landmarks_letter_hands_only_mlp.pkl'
OUT = 'results/ISL_MLP'

with open(SRC, 'rb') as f:
    dataset = pickle.load(f)

dataset['landmarks'] = [np.asarray(frame)[:42] for frame in dataset['landmarks']]
dataset['input_features'] = 126
dataset['mode'] = 'hands_only_mlp'
Path(DST).parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'wb') as f:
    pickle.dump(dataset, f)
print(f'Sliced {len(dataset["landmarks"])} samples to hands-only (126 features) -> {DST}')

trainer = ISLTrainer('config.yaml')
cfg = trainer.config.config
cfg['model']['type'] = 'MLP'
cfg['model']['sequence_length'] = 1
cfg['model']['input_features'] = 126
cfg['paths']['model_save_dir'] = OUT
cfg['paths']['logs_dir'] = f'{OUT}/logs'
cfg['paths']['checkpoints_dir'] = f'{OUT}/checkpoints'

history, accuracy = trainer.train(DST, time_budget_minutes=30)
print(f'ISL_MLP_EPOCHS: {len(history.history["loss"])}')
print(f'ISL_MLP_TEST_ACCURACY: {accuracy}')
