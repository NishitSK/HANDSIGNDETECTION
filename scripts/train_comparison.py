"""
Multi-architecture training comparison for the ISL letter classifier.

Reuses the landmarks already extracted from the RealSign ISL dataset
(data/landmarks/landmarks_letter_both_hands_face.pkl) and trains LSTM, GRU,
TCN and TRANSFORMER under a shared, time-boxed budget, so the four are
compared fairly (same data split, same seed, same held-out test set) rather
than each being tuned/babysat individually.

Two phases:
  1. A short fixed-length benchmark of every architecture, to measure real
     epochs-per-minute on this machine (informed allocation beats guessing).
  2. A full run per architecture, capped by BOTH a wall-clock time budget
     (TimeBudgetCallback) and the usual epoch/early-stopping config, so no
     single architecture can consume the whole remaining budget.

Every run's outputs (model, metadata, plots, classification report) are
copied to results/<ARCH>/ so they survive the next architecture's run
overwriting the shared models/saved/ and logs/ paths. At the end, the best
architecture by held-out TEST accuracy is promoted to models/saved/isl_model.h5
and config.yaml is updated to match.
"""
import json
import shutil
import time
import traceback
from pathlib import Path

import yaml

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ISLTrainer

LANDMARKS_PATH = 'data/landmarks/landmarks_letter_both_hands_face.pkl'
CONFIG_PATH = 'config.yaml'
ARCHITECTURES = ['LSTM', 'GRU', 'TCN', 'TRANSFORMER']
RESULTS_DIR = Path('results')

BENCHMARK_MINUTES = 2
TOTAL_BUDGET_MINUTES = 8 * 60
FINAL_BUFFER_MINUTES = 30
FLOOR_MINUTES_PER_ARCH = 60
MAX_EXTRA_SHARE = 0.5  # no single architecture gets more than this share of the speed-weighted pool

LOG_FILES = [
    'confusion_matrix.png',
    'per_class_accuracy.png',
    'classification_report.txt',
    'confidence_distribution.png',
    'training_history.png',
]


def copy_run_outputs(arch, tag):
    """Copy this run's model + evaluation artifacts into results/<arch>/<tag>/."""
    dest = RESULTS_DIR / arch / tag
    dest.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in [
        ('models/saved/isl_model.h5', 'isl_model.h5'),
        ('models/saved/model_metadata.json', 'model_metadata.json'),
    ]:
        src = Path(src_name)
        if src.exists():
            shutil.copy(src, dest / dst_name)

    logs_dir = Path('logs')
    for fname in LOG_FILES:
        src = logs_dir / fname
        if src.exists():
            shutil.copy(src, dest / fname)

    training_log = logs_dir / f'{arch}_training_log.csv'
    if training_log.exists():
        shutil.copy(training_log, dest / 'training_log.csv')

    return dest


def run_one(trainer, arch, time_budget_minutes):
    """Run a single architecture; returns a result dict (never raises)."""
    print("\n" + "#" * 70)
    print(f"# {arch}  (budget: {time_budget_minutes:.1f} min)")
    print("#" * 70 + "\n")

    trainer.config.config['model']['type'] = arch

    start = time.time()
    try:
        history, test_accuracy = trainer.train(
            LANDMARKS_PATH,
            time_budget_minutes=time_budget_minutes,
        )
        elapsed_min = (time.time() - start) / 60
        epochs_completed = len(history.history.get('loss', []))
        n_params = trainer.model.count_params()
        error = None
    except Exception as e:
        elapsed_min = (time.time() - start) / 60
        test_accuracy = None
        epochs_completed = 0
        n_params = None
        error = f"{type(e).__name__}: {e}"
        print(f"\n[ERROR] {arch} run failed after {elapsed_min:.1f} min: {error}")
        traceback.print_exc()

    result = {
        'architecture': arch,
        'test_accuracy': test_accuracy,
        'epochs_completed': epochs_completed,
        'elapsed_minutes': round(elapsed_min, 2),
        'budget_minutes': round(time_budget_minutes, 2),
        'params': n_params,
        'error': error,
    }

    if error is None:
        copy_run_outputs(arch, 'final')

    return result


def main():
    with open(CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    trainer = ISLTrainer(CONFIG_PATH)

    # ---- Phase 1: short benchmark of every architecture ----
    print("\n" + "=" * 70)
    print(f"BENCHMARK PHASE — {BENCHMARK_MINUTES} min per architecture")
    print("=" * 70)

    benchmark = {}
    for arch in ARCHITECTURES:
        r = run_one(trainer, arch, BENCHMARK_MINUTES)
        benchmark[arch] = r
        if r['error'] is None:
            copy_run_outputs(arch, 'benchmark')
        print(f"[BENCHMARK] {arch}: {r['epochs_completed']} epoch(s) in "
              f"{r['elapsed_minutes']:.1f} min, test_acc={r['test_accuracy']}")

    # ---- Compute time allocation for the real runs ----
    spent_so_far = sum(r['elapsed_minutes'] for r in benchmark.values())
    remaining = TOTAL_BUDGET_MINUTES - spent_so_far - FINAL_BUFFER_MINUTES
    remaining = max(remaining, FLOOR_MINUTES_PER_ARCH * len(ARCHITECTURES))

    pool_after_floors = max(remaining - FLOOR_MINUTES_PER_ARCH * len(ARCHITECTURES), 0)

    speed = {}
    for arch, r in benchmark.items():
        if r['error'] is not None:
            speed[arch] = 0.0
        elif r['epochs_completed'] > 0:
            speed[arch] = r['epochs_completed'] / r['elapsed_minutes']
        else:
            # Didn't even finish one epoch in the benchmark window — treat as
            # slow but not zero, so it still gets a share of the pool.
            speed[arch] = 0.25 / r['elapsed_minutes']

    total_speed = sum(speed.values()) or 1.0
    allocation = {}
    for arch in ARCHITECTURES:
        share = min(speed[arch] / total_speed, MAX_EXTRA_SHARE)
        allocation[arch] = FLOOR_MINUTES_PER_ARCH + share * pool_after_floors

    # Renormalize so the extra pool is fully used despite the per-arch cap
    allocated_extra = sum(allocation[a] - FLOOR_MINUTES_PER_ARCH for a in ARCHITECTURES)
    leftover = pool_after_floors - allocated_extra
    if leftover > 0:
        uncapped = [a for a in ARCHITECTURES if speed[a] / total_speed < MAX_EXTRA_SHARE]
        if uncapped:
            bonus = leftover / len(uncapped)
            for a in uncapped:
                allocation[a] += bonus

    print("\n" + "=" * 70)
    print("TIME ALLOCATION FOR FULL RUNS")
    print("=" * 70)
    for arch in ARCHITECTURES:
        print(f"  {arch:12s}: {allocation[arch]:6.1f} min "
              f"(benchmark speed: {speed[arch]:.3f} epochs/min)")
    print(f"  (spent on benchmarking: {spent_so_far:.1f} min, "
          f"reserved buffer: {FINAL_BUFFER_MINUTES} min)")

    # ---- Phase 2: full runs ----
    print("\n" + "=" * 70)
    print("FULL TRAINING RUNS")
    print("=" * 70)

    final_results = []
    for arch in ARCHITECTURES:
        r = run_one(trainer, arch, allocation[arch])
        final_results.append(r)

    # ---- Pick winner, promote it, write comparison report ----
    successful = [r for r in final_results if r['error'] is None and r['test_accuracy'] is not None]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / 'comparison_results.json', 'w') as f:
        json.dump({'benchmark': benchmark, 'final': final_results, 'allocation': allocation}, f, indent=2)

    if not successful:
        print("\n[FATAL] Every architecture failed. See results/comparison_results.json for errors.")
        return

    winner = max(successful, key=lambda r: r['test_accuracy'])
    print("\n" + "=" * 70)
    print(f"WINNER: {winner['architecture']} — test accuracy {winner['test_accuracy']:.4f}")
    print("=" * 70)

    winner_dir = RESULTS_DIR / winner['architecture'] / 'final'
    shutil.copy(winner_dir / 'isl_model.h5', 'models/saved/isl_model.h5')
    shutil.copy(winner_dir / 'model_metadata.json', 'models/saved/model_metadata.json')

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['model']['type'] = winner['architecture']
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nPromoted {winner['architecture']} to models/saved/isl_model.h5")
    print(f"config.yaml model.type set to {winner['architecture']}")
    print("\nDone.")


if __name__ == '__main__':
    main()
