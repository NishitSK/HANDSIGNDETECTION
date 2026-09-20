# Manual Training Guide - Complete Workflow

## 🎯 Why Manual Collection is Better

**Automatic detection problems:**
- ❌ May capture wrong gestures
- ❌ No control over timing
- ❌ Hard to get consistent data
- ❌ Can't handle complex phrases

**Manual collection benefits:**
- ✅ **100% accuracy** - you control what's captured
- ✅ **Dynamic gestures** - record motion-based signs
- ✅ **Multi-sign phrases** - combine multiple gestures
- ✅ **Quality control** - review before saving
- ✅ **Flexible** - photos for static, videos for dynamic

---

## 📸 Photo vs Video - When to Use Each

### Use **PHOTOS** for:
- ✅ **Single letters** (A-Z)
- ✅ **Numbers** (0-9)
- ✅ **Static gestures** - single hand position
- ✅ **Quick collection** - 50-100 photos per class

**Examples:** 
- Alphabet letters: A, B, C, D...
- Numbers: 1, 2, 3, 4...
- Simple signs: "Yes", "No", "Stop"

### Use **VIDEOS** for:
- ✅ **Motion-based signs** - require movement
- ✅ **Dynamic gestures** - changing hand positions
- ✅ **Phrases** - multiple signs in sequence
- ✅ **Complex signs** - need context of movement

**Examples:**
- "How are you" (3 separate signs)
- "Thank you" (motion from chin outward)
- "Sorry" (circular motion on chest)
- "Please" (rubbing motion)

---

## 🚀 Quick Start Guide

### Step 1: Launch Data Collection Tool

```bash
python collect_training_data.py
```

### Step 2: Set Your First Class

1. Press **C** to change class
2. Enter class name (e.g., "A" or "Hello" or "How are you")
3. Press Enter

### Step 3: Choose Mode

- Press **M** to toggle between Photo/Video mode
- 📸 **Photo Mode** - for static gestures
- 🎥 **Video Mode** - for dynamic gestures

### Step 4: Collect Data

**For Photos:**
1. Make your gesture
2. Hold it steady
3. Press **SPACE** to capture
4. Repeat 50-100 times from different angles

**For Videos:**
1. Press **R** to start recording
2. Perform the gesture (2-3 seconds)
3. Press **R** to stop
4. Repeat 10-20 times

### Step 5: Save Session

- Press **S** to save your session
- Data saved to `data/collected_data/`

---

## 📝 Collection Strategy

### For Alphabets (A-Z):
```
Goal: 100 photos per letter
Mode: Photo
Time: ~30 minutes for all 26 letters

Tips:
- Vary hand angles (front, side, tilted)
- Different distances from camera
- Different lighting conditions
- Both hands if letter uses both
```

### For Numbers (0-9):
```
Goal: 80 photos per number
Mode: Photo
Time: ~15 minutes for all 10 numbers

Tips:
- Clear finger positions
- Both left and right hand versions
- Different orientations
```

### For Simple Words (Static):
```
Examples: "Yes", "No", "Stop", "Hello"
Goal: 60 photos per word
Mode: Photo
Time: ~10 minutes per word

Tips:
- Include facial expressions
- Natural hand positions
- Both hands visible
```

### For Motion-Based Signs:
```
Examples: "Thank you", "Sorry", "Please"
Goal: 15 videos per sign
Mode: Video
Time: ~5 minutes per sign

Tips:
- Complete the full motion
- 2-3 seconds per video
- Smooth, natural movement
- Start and end clearly
```

### For Phrases (Multi-Sign):
```
Examples: "How are you", "What is your name"
Goal: 20 videos per phrase
Mode: Video
Time: ~10 minutes per phrase

Tips:
- Perform each sign in sequence
- Brief pause between signs
- 5-8 seconds per video
- Natural flow, not robotic
```

---

## 🎮 Keyboard Controls

| Key | Action | Mode |
|-----|--------|------|
| **SPACE** | Capture photo | Photo mode |
| **R** | Start/Stop recording | Video mode |
| **C** | Change class name | Both |
| **M** | Switch Photo/Video mode | Both |
| **S** | Save session | Both |
| **H** | Show help | Both |
| **Q** | Quit and save | Both |

---

## 💾 File Structure

After collection, your data will be organized like this:

```
data/collected_data/
├── A/
│   ├── photos/
│   │   ├── 20251104_120530_001.jpg
│   │   ├── 20251104_120530_001_landmarks.npy
│   │   └── ...
│   └── videos/
│       ├── 20251104_121045.avi
│       ├── 20251104_121045_landmarks.pkl
│       └── ...
├── B/
├── Thank you/
│   └── videos/
│       └── ... (motion-based, so videos)
├── How are you/
│   └── videos/
│       └── ... (phrase, so videos)
└── session_20251104_120000.json
```

---

## 🏋️ Training After Collection

### Step 1: Prepare Training Script

Edit `train.py` and set:

```python
# For letters
TRAIN_TYPE = "letter"
CLASSES = ['A', 'B', 'C', ...]  # Classes you collected

# For phrases
TRAIN_TYPE = "phrase"
CLASSES = ['Thank you', 'How are you', ...]

# For mixed
TRAIN_TYPE = "custom"
CLASSES = ['A', 'B', 'Hello', 'Thank you', ...]
```

### Step 2: Extract Landmarks

The tool already saves landmarks automatically!
- Photos: `_landmarks.npy` files
- Videos: `_landmarks.pkl` files

### Step 3: Train Model

```bash
# Update train.py with your classes
python train.py
```

### Step 4: Test Your Model

```bash
# Use manual capture mode
python main.py
# Select option 1 (GUI)
# Press SPACE to capture your gestures
```

---

## 🎯 Best Practices

### 1. **Quality Over Quantity**
- 50 good photos > 200 bad photos
- Ensure hands/face clearly visible
- Good lighting is essential

### 2. **Variety is Key**
- Different hand angles
- Different distances
- Different lighting
- Different backgrounds

### 3. **Consistency Matters**
- Same gesture should look similar
- Don't mix different variations
- Review captured data regularly

### 4. **For Phrases:**
- **Option A: Single Video** - Record entire phrase as one video
  * Pro: Natural flow
  * Con: Harder to train
  
- **Option B: Separate Videos** - Train each word separately, combine in app
  * Pro: Easier to train, reusable
  * Con: Less natural flow
  
- **Recommended:** Use Option B - Train "How", "are", "you" separately, then the app combines them

### 5. **Test Frequently**
- Collect 20 samples → Test
- If accuracy < 80% → Collect more varied data
- If accuracy > 90% → Move to next class

---

## 📊 Recommended Collection Plan

### Day 1: Alphabets (2 hours)
- Collect 100 photos per letter (A-Z)
- Total: 2,600 photos
- Use Photo mode

### Day 2: Numbers (1 hour)
- Collect 80 photos per number (0-9)
- Total: 800 photos
- Use Photo mode

### Day 3: Common Words (2 hours)
- Static words: "Yes", "No", "Hello", "Stop", "Help", "Please"
- 60 photos each
- Total: 360 photos
- Use Photo mode

### Day 4: Motion Signs (2 hours)
- Dynamic signs: "Thank you", "Sorry", "Welcome", "Good"
- 15 videos each
- Total: 60 videos
- Use Video mode

### Day 5: Phrases (3 hours)
- Common phrases: "How are you", "What is your name", "Nice to meet you"
- 20 videos each
- Total: 60 videos
- Use Video mode

---

## 🔧 Troubleshooting

### "No hands/face detected"
- ✅ Ensure good lighting
- ✅ Keep hands in frame
- ✅ Move closer to camera
- ✅ Clean camera lens

### "Too many/few landmarks"
- ✅ Check if both hands visible when needed
- ✅ Ensure face is in frame
- ✅ Adjust detection confidence in code

### "Video recording slow"
- ✅ Reduce frame rate
- ✅ Smaller resolution
- ✅ Shorter videos (2-3 seconds max)

### "Training not working"
- ✅ Ensure consistent data
- ✅ Check class names match exactly
- ✅ Verify landmark files exist
- ✅ Try with fewer classes first

---

## 🎓 Pro Tips

1. **Start Small**: Begin with 5 letters, perfect them, then expand
2. **Test Early**: Don't collect 26 letters before testing
3. **Use Both Hands**: Some signs need both - ensure both visible
4. **Include Face**: Facial expressions matter in ISL
5. **Natural Movement**: For videos, don't be robotic
6. **Lighting**: Consistent, bright lighting is crucial
7. **Background**: Plain background helps detection
8. **Sessions**: Save frequently - don't lose hours of work

---

## ✅ Success Checklist

Before training, ensure:

- [ ] Collected data for all planned classes
- [ ] Minimum 50 samples per class (photos or videos)
- [ ] Variety in angles and positions
- [ ] Landmarks saved correctly (.npy or .pkl files)
- [ ] Session saved (JSON file exists)
- [ ] Tested on a small subset first
- [ ] Class names match in train.py

---

**Ready to start?**

```bash
python collect_training_data.py
```

**Need help?** Press **H** in the application!
