# Indian Sign Language (ISL) Detection & Translation System

## 🎯 Project Overview

An advanced real-time Indian Sign Language detection and translation system with grammar correction and voice synthesis, designed for professional use including news channels and formal communications.

## ✨ Key Features

- **Real-time ISL Detection**: Fast and accurate hand gesture recognition using custom deep learning models
- **99% Accuracy Target**: Advanced LSTM/Transformer models trained on extensive ISL datasets
- **MediaPipe Integration**: Manual training using MediaPipe hand landmarks for robust feature extraction
- **Grammar Correction**: AI-powered sentence formation for grammatically correct output
- **Voice Synthesis**: Natural text-to-speech conversion for seamless communication
- **Professional GUI**: User-friendly interface suitable for news channels and professional settings
- **High Performance**: Optimized for real-time processing at 30+ FPS

## 🏗️ Architecture

```
├── data/                      # Dataset and processed data
│   ├── ISL_dataset/          # Raw ISL video/image data
│   ├── processed/            # Preprocessed data
│   └── landmarks/            # Extracted MediaPipe landmarks
├── models/                    # Model architectures and saved weights
│   ├── gesture_model.py      # LSTM/Transformer model
│   ├── saved/                # Trained model weights
│   └── checkpoints/          # Training checkpoints
├── src/                       # Source code
│   ├── data_collection.py    # Data gathering and preprocessing
│   ├── landmark_extraction.py # MediaPipe landmark extraction
│   ├── train.py              # Model training script
│   ├── inference.py          # Real-time inference engine
│   ├── grammar_correction.py # NLP-based sentence formation
│   ├── tts_engine.py         # Text-to-speech module
│   └── gui_app.py            # Main GUI application
├── utils/                     # Utility functions
│   ├── data_augmentation.py  # Data augmentation
│   ├── visualization.py      # Metrics and result visualization
│   └── config_loader.py      # Configuration management
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── main.py                   # Main application entry point
```

## 🚀 Installation

### Prerequisites
- Python 3.8 - 3.10
- Webcam for real-time detection
- 8GB+ RAM recommended
- GPU (CUDA-compatible) recommended for training

### Setup Steps

1. **Clone/Navigate to project directory**
```bash
cd "c:\Users\hemanth\Desktop\mini_project 22"
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download ISL Dataset** (Multiple sources)
   - Indian Sign Language Dataset (Kaggle)
   - INCLUDE Dataset
   - Custom collected data

## 📊 Dataset Information

The currently trained model uses the [RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)
(CC0-1.0): 26 fingerspelled ISL letter classes (A–Z), ~1,000 real photos per
class from four signers with varied lighting/skin tone/pose, ~26,000 images
total. Extracted to `data/RealSign_ISL/<letter>/` (gitignored — download
separately; see `RESULTS_REPORT.md` for the exact source and license).

Note: ISL fingerspelling is not one-handed the way ASL is — several letters
(e.g. 'A') use both hands, which is why this pipeline runs in "both hands +
face" landmark mode rather than a single-hand mode.

Full methodology, bugs found/fixed, and the architecture comparison that
produced the current model are documented in **`RESULTS_REPORT.md`**.

## 🎓 Training the Model

The unified entry point is `train.py` (project root) — it extracts MediaPipe
landmarks from `data/RealSign_ISL/` (path configurable via `config.yaml`'s
`data.dataset_path`), caches them to `data/landmarks/`, and trains via
`src/train.py`'s `ISLTrainer`:

```bash
python train.py
```

Edit the `TRAIN_TYPE`/`TRAINING_MODE`/`EPOCHS` constants near the top of
`train.py` to retrain on different classes or settings. To compare multiple
architectures (LSTM/GRU/TCN/Transformer) under a shared time budget the way
the current model was chosen, see `scripts/train_comparison.py`.

### Evaluate an existing model
```bash
python src/train.py --evaluate --model models/saved/isl_model.h5 --data data/landmarks/landmarks_letter_both_hands_face.pkl
```

## 🎬 Running the Application

### GUI Application (Recommended)
```bash
python main.py
```

### Command Line Inference
```bash
python src/inference.py --model models/saved/best_model.h5 --camera 0
```

### Mobile (phone browser)

`mobile/` is a lightweight version that runs entirely on the phone: MediaPipe Web
finds the hand landmarks, a small single-frame MLP model runs in the browser
through TensorFlow.js, and the phone's own text-to-speech speaks the sentence.
Supports both **ISL** (two-handed) and **ASL** (one-handed) fingerspelling as a
mode switch in Settings — each is its own 160 KB model, loaded on demand. The
948 MB Flan-T5 model is left out — the desktop app loads it but never uses it,
and the rule-based grammar it actually uses is ported to `mobile/app/grammar.mjs`.

An earlier version used the desktop TCN model (hands+face, 2.4 MB) directly.
Real-world testing on a phone found classification "really slow" and "not
usable" — the TCN's attention layer has no accelerated fallback on phones with
a weak or missing WebGL driver, so it fell back to ~5.4s/prediction on plain
CPU. The current hands-only MLP has no such layer, runs on a `webgl → wasm →
cpu` fallback chain, and predicts in single-digit milliseconds even on the
slowest of those three. Full rationale and numbers: `RESULTS_REPORT.md` §8.6.

1. On this PC, with the phone on the same Wi-Fi:
   ```bash
   python mobile/serve_https.py
   ```
2. On the phone, open the `https://…:8443` address it prints, accept the
   certificate warning (the server uses a self-signed certificate for local
   testing), and allow the camera.
3. If the phone can't connect, allow Python through Windows Firewall for
   private networks.

The first visit downloads the tracking runtime and both mode models
(under 1 MB combined for the models; the MediaPipe hand-tracking runtime is
the bulk of the download) from jsDelivr and Google. A service worker caches
everything after that, so later visits don't re-download it.

Using it: pick ISL or ASL in Settings, hold a letter steady and it's added
automatically (drop your hand briefly to repeat a letter), tap **Space**
between words, and **Speak** to hear the sentence. The **Letters** button
shows a reference photo for every letter the current mode supports (ASL has
no J/Z — see below). Settings also has a front/back camera switch and a
**Mirror camera input** option to try if letters are often misread.

How well it works: on held-out test samples, the browser model's WebGL
predictions match Keras's argmax 100% of the time for both modes (ISL: 4,396
samples, 98.27% Keras accuracy; ASL: 444 samples, 92.34% Keras accuracy — ASL's
lower number is dataset size, ~90 photos/letter vs ISL's much larger set, not
the architecture). J and Z are motion signs with no static handshape; ISL's
source photos happened to include static frames for them (trained and
labeled "motion sign — still frame" in the pamphlet), ASL's source dataset has
none at all, so the ASL model simply has no J/Z classes. `RESULTS_REPORT.md`
§8.6 has the full writeup, including a verification-tooling bug the ASL
conversion caught (a hardcoded raw-feature stride that happened to work for
ISL by coincidence).

#### Converting a retrained model for the phone

Conversion needs its own environment (TensorFlow 2.15 + tensorflowjs 4.17),
separate from the training venv. On Windows, tensorflowjs imports JAX and
TensorFlow Decision Forests at load time even though this conversion never uses
them; JAX installs normally, and Decision Forests (which has no Windows build) is
replaced by an empty package:

```bash
py -3.11 -m venv ../.venv-convert
```
```bash
../.venv-convert/Scripts/python.exe -m pip install "tensorflow==2.15.0" "tf-keras==2.15.0" "tensorflow-hub>=0.14.0" "importlib_resources>=5.9.0" six "packaging~=23.1" h5py "jax[cpu]==0.4.23" "numpy<2" "ml-dtypes~=0.2.0"
```
```bash
../.venv-convert/Scripts/python.exe -m pip install "tensorflowjs==4.17.0" --no-deps
```
```bash
../.venv-convert/Scripts/python.exe -c "import site, pathlib; p = pathlib.Path(site.getsitepackages()[-1]) / 'tensorflow_decision_forests'; p.mkdir(exist_ok=True); (p / '__init__.py').touch()"
```

Then convert, pointing `--out` at the mode's own folder (float16 weights):

```bash
../.venv-convert/Scripts/python.exe mobile/convert_model.py results/ISL_MLP/isl_model.h5 --out mobile/app/model/isl
../.venv-convert/Scripts/python.exe mobile/convert_model.py results/ASL_MLP/isl_model.h5 --out mobile/app/model/asl
```

Add a `"num_hands"` field to the mode's `metadata.json` afterward if it isn't
already there (2 for ISL, 1 for ASL) — the app reads it to size the hand
tracker.

To check a conversion, first export the held-out test split and Keras's
predictions on it (training venv). `--seq-len 1` is for the current
single-frame MLP models (`--seq-len 30` is only for the retired TCN). Point
`--out-dir` at a mode-specific folder so ISL and ASL exports don't collide:

```bash
python mobile/tests/export_reference.py --model results/ISL_MLP/isl_model.h5 --features 126 --seq-len 1 --out-dir results/mobile_verification
python mobile/tests/export_reference.py --model results/ASL_MLP/isl_model.h5 --features 126 --seq-len 1 --landmarks data/landmarks/landmarks_asl_hands_only_mlp.pkl --out-dir results/mobile_verification_asl
```

Then serve the project on this PC only and open the check pages in a desktop browser:

```bash
python mobile/serve_https.py --test
```

- `http://localhost:8082/mobile/tests/parity.html?model=../app/model/asl/model.json&data=../../results/mobile_verification_asl/&features=126`
  scores a converted model on WebGL (the backend phones use) against Keras on
  every held-out sample (omit `?model=`/`?data=` for the ISL defaults).
- `http://localhost:8082/mobile/tests/mediapipe_parity.html` runs MediaPipe Web on
  held-out photos and compares its landmarks and predicted letters with the Python
  pipeline the model was trained on. ISL only — this needs each test sample's
  source photo path recoverable from its group id, which the ASL landmarks
  pickle doesn't carry.

`mobile/tests/verify_pipeline.mjs` runs the model check in Node, but only on the
slow CPU backend; use `--limit` for a quick look.

## 🎯 Usage Guide

1. **Launch Application**: Run `python main.py`
2. **Camera Setup**: Position yourself in front of the camera with good lighting
3. **Perform Gestures**: Make ISL signs clearly within the camera frame
4. **Real-time Translation**: See detected signs, corrected sentences, and hear voice output
5. **Recording Mode**: Enable recording for news channel broadcasts

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture (LSTM/Transformer)
- Detection thresholds
- TTS settings
- Camera parameters
- GUI preferences

## 📈 Performance Metrics

Current model (TCN, 26 ISL letters, see `RESULTS_REPORT.md` for full methodology):
- **Held-out test accuracy**: 98.89% (4,396 test images, never used in training/model selection)
- **Model size**: 1.2M parameters (smallest of 4 architectures compared, also the most accurate)
- **Per-class precision/recall**: consistently mid-90s–100% across all 26 letters

Not yet measured in this environment (no webcam/GPU available for live testing):
- **FPS** / **Latency**: real-time performance target is 30+ FPS / <100ms, per the original design goal — untested live
- **Grammar Accuracy**: rule-based grammar correction is implemented (`src/grammar_correction.py`) but not benchmarked against a labeled set

## 🛠️ Advanced Features

### 1. Grammar Correction
Rule-based sentence formation (`src/grammar_correction.py`) turns sign sequences into sentences:
```
Input:  "who you"
Output: "Who are you?"
```
An optional Flan-T5 path exists (`correct_sentence(..., use_model=True)`) but is off by default.

### 2. Adaptive Smoothing
Temporal smoothing reduces jitter and false positives in real-time detection.

### 3. Multi-modal Output
- Visual: On-screen text display
- Audio: Natural voice synthesis
- Logging: Conversation history

## 🎥 News Channel Features

- **Professional UI**: Clean interface suitable for broadcast
- **High Confidence Display**: Shows detection confidence
- **Recording Capability**: Save translation sessions
- **Low Latency**: Real-time with minimal delay
- **Grammatical Output**: Properly structured sentences

## 🤝 Contributing

This is a main project. Future enhancements:
- More ISL signs (expand to 500+ vocabulary)
- Regional ISL variations
- Two-hand gesture support
- Contextual understanding

## 📝 License

Educational Project - Indian Sign Language Recognition System

## 👥 Contact

For questions or issues related to this project, please refer to the documentation.

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- Transformers by Hugging Face for NLP
- ISL dataset contributors
- Indian Sign Language research community

---

**Note**: This system is designed for Indian Sign Language (ISL). Training data and model weights need to be obtained/trained before use.
