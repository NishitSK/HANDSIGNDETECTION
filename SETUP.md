# Quick Setup Guide

## 🚀 Quick Start (5 Steps)

### 1. Install Dependencies
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Setup Dataset Structure
```powershell
python src/data_collection.py --setup
```

### 3. Download ISL Dataset

**Option A: Kaggle Datasets**
- Visit: https://www.kaggle.com/search?q=indian+sign+language
- Download "Indian Sign Language Dataset" 
- Extract to `data/ISL_dataset/`

**Option B: INCLUDE Dataset**
- Visit: https://zenodo.org/record/4010759
- Download and extract to `data/ISL_dataset/`

**Option C: Record Your Own**
```powershell
python src/data_collection.py --record --sign "Hello" --samples 100
```

### 4. Extract Landmarks & Train Model
```powershell
# Extract hand landmarks from videos/images
python src/landmark_extraction.py --input data/ISL_dataset --output data/landmarks

# Train the model (this will take time depending on dataset size)
python src/train.py --epochs 200
```

### 5. Run the Application
```powershell
# GUI Application (Recommended)
python main.py

# Or directly:
python src/gui_app.py

# Command-line version
python src/inference.py --model models/saved/isl_model.h5
```

---

## 📋 Detailed Workflow

### Data Collection Phase

1. **Setup dataset structure** (100 ISL signs):
   ```powershell
   python src/data_collection.py --setup
   ```

2. **Populate with data**: Add videos/images to each sign folder
   - Target: 100+ samples per sign
   - Format: MP4, AVI, or images
   - Quality: Clear hand visibility

3. **Validate dataset**:
   ```powershell
   python src/data_collection.py --validate
   ```

### Training Phase

1. **Extract landmarks**:
   ```powershell
   python src/landmark_extraction.py
   ```

2. **Train model**:
   ```powershell
   python src/train.py --epochs 200 --batch-size 32
   ```

3. **Monitor training**:
   - Check `logs/` folder for visualizations
   - View training metrics in `logs/training_log.csv`
   - TensorBoard: `tensorboard --logdir logs/`

### Testing Phase

1. **Evaluate model**:
   ```powershell
   python src/train.py --evaluate --model models/saved/isl_model.h5
   ```

2. **Test real-time**:
   ```powershell
   python src/inference.py
   ```

3. **Launch GUI**:
   ```powershell
   python main.py
   ```

---

## 🎯 Expected Results

- **Accuracy**: 95-99% on test set
- **FPS**: 25-30 frames per second
- **Latency**: < 100ms per prediction
- **Confidence**: > 85% for correct predictions

---

## 🔧 Troubleshooting

### Package Import Errors
```powershell
pip install --upgrade -r requirements.txt
```

### Camera Not Working
```powershell
# Test camera ID
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Low Accuracy
- Collect more data (target: 100+ samples per sign)
- Increase training epochs
- Check data quality (lighting, hand visibility)
- Enable data augmentation

### GPU Not Detected
```powershell
# Install CUDA-enabled TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0
```

---

## 📊 Project Structure

```
mini_project 22/
├── main.py                 # Main entry point
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
│
├── data/                 # Dataset
│   ├── ISL_dataset/     # Raw videos/images
│   ├── landmarks/       # Extracted landmarks
│   └── processed/       # Preprocessed data
│
├── models/              # Model files
│   ├── gesture_model.py # Model architecture
│   └── saved/          # Trained models
│
├── src/                 # Source code
│   ├── data_collection.py
│   ├── landmark_extraction.py
│   ├── train.py
│   ├── inference.py
│   ├── grammar_correction.py
│   ├── tts_engine.py
│   └── gui_app.py
│
├── utils/               # Utilities
│   ├── config_loader.py
│   ├── data_augmentation.py
│   └── visualization.py
│
└── logs/                # Training logs
```

---

## 💡 Tips for 99% Accuracy

1. **Large Dataset**: 100+ samples per sign
2. **Data Quality**: Good lighting, clear hand visibility
3. **Variety**: Different people, angles, speeds
4. **Data Augmentation**: Enabled by default
5. **Training Time**: 200+ epochs recommended
6. **Model Architecture**: LSTM (default) or Transformer
7. **Fine-tuning**: Adjust learning rate, dropout

---

## 🎥 News Channel Setup

For professional news channel use:

1. **Good Lighting**: Ensure bright, even lighting
2. **Clean Background**: Solid color background preferred
3. **Camera Position**: Eye level, arm's length distance
4. **Settings**: 
   - Enable "Formal Mode" in grammar correction
   - Adjust TTS rate to 140-150 WPM
   - Use professional female voice

---

## 📞 Need Help?

Check the main README.md for comprehensive documentation.
```

