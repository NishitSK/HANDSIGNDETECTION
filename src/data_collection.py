"""
Data Collection and Download Utility
Downloads ISL datasets and prepares data for training
"""

import os
import requests
import zipfile
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json


class ISLDatasetCollector:
    """Collect and organize ISL dataset"""
    
    def __init__(self, base_dir='data/ISL_dataset'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset_structure(self):
        """
        Create a sample dataset structure for demonstration
        Users should populate this with actual ISL data
        """
        # Common ISL signs to include
        isl_signs = [
            'Hello', 'Thank_You', 'Please', 'Sorry', 'Yes', 'No',
            'Good', 'Bad', 'Help', 'Water', 'Food', 'Eat', 'Drink',
            'Morning', 'Evening', 'Night', 'Day', 'Week', 'Month', 'Year',
            'Today', 'Tomorrow', 'Yesterday', 'Now', 'Later',
            'I', 'You', 'He', 'She', 'We', 'They',
            'What', 'Where', 'When', 'Why', 'How', 'Who',
            'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
            'Happy', 'Sad', 'Angry', 'Tired', 'Sick', 'Fine',
            'Home', 'School', 'Work', 'Hospital', 'Shop', 'Office',
            'Mother', 'Father', 'Brother', 'Sister', 'Friend', 'Family',
            'Name', 'Age', 'Time', 'Date', 'Place', 'Person',
            'Read', 'Write', 'Listen', 'See', 'Speak', 'Understand',
            'Come', 'Go', 'Sit', 'Stand', 'Walk', 'Run',
            'Hot', 'Cold', 'Big', 'Small', 'Fast', 'Slow',
            'News', 'Important', 'Information', 'Question', 'Answer',
            'Weather', 'Rain', 'Sun', 'Wind', 'Cloud',
            'Country', 'City', 'India', 'Government', 'People',
            'Money', 'Price', 'Buy', 'Sell', 'Pay'
        ]
        
        print(f"Creating dataset structure for {len(isl_signs)} ISL signs...")
        
        for sign in tqdm(isl_signs, desc="Creating folders"):
            sign_dir = self.base_dir / sign
            sign_dir.mkdir(exist_ok=True)
        
        # Create README for data collection
        readme_content = """# ISL Dataset Collection Guide

## Dataset Structure

Each folder represents one ISL sign. Populate each folder with:
- Videos (.mp4, .avi) of the sign being performed
- Multiple samples from different people
- Different lighting conditions and backgrounds
- Target: 100+ samples per sign for 99% accuracy

## Data Collection Sources

### 1. Online Datasets
- **INCLUDE Dataset**: Indian Sign Language dataset
  - Download from: https://zenodo.org/record/4010759
  
- **Kaggle ISL Datasets**:
  - Search "Indian Sign Language" on Kaggle
  - Download and organize by sign name

### 2. Manual Recording
Use the built-in recording utility:
```python
python src/data_collection.py --record --sign "Hello" --samples 100
```

### 3. Data Requirements
- **Quality**: Clear hand visibility, good lighting
- **Variety**: Different people, angles, speeds
- **Duration**: 2-5 seconds per video
- **Format**: MP4 or AVI preferred
- **Resolution**: 640x480 minimum

### 4. Naming Convention
- Videos: sign_name_001.mp4, sign_name_002.mp4, etc.
- Keep consistent naming for easier processing

## Current Status
Total Signs: """ + str(len(isl_signs)) + """
Target Samples per Sign: 100
Total Target Samples: """ + str(len(isl_signs) * 100) + """

## Next Steps
1. Download datasets from sources above
2. Organize videos into respective folders
3. Run landmark extraction: `python src/landmark_extraction.py`
4. Train model: `python src/train.py`
"""
        
        with open(self.base_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        # Save sign list
        with open(self.base_dir / 'sign_list.json', 'w') as f:
            json.dump({'signs': isl_signs, 'count': len(isl_signs)}, f, indent=2)
        
        print(f"\n✓ Dataset structure created at: {self.base_dir}")
        print(f"✓ Total sign categories: {len(isl_signs)}")
        print(f"\nIMPORTANT: Populate folders with ISL videos/images")
        print(f"Target: 100+ samples per sign for optimal accuracy")
        print(f"\nSee {self.base_dir}/README.md for collection guide")
        
        return isl_signs
    
    def record_sign_samples(self, sign_name, num_samples=10, duration_sec=3):
        """
        Record sign samples using webcam
        
        Args:
            sign_name: Name of the sign to record
            num_samples: Number of samples to record
            duration_sec: Duration of each sample in seconds
        """
        sign_dir = self.base_dir / sign_name
        sign_dir.mkdir(exist_ok=True)
        
        # Get next sample number
        existing_files = list(sign_dir.glob('*.mp4'))
        start_idx = len(existing_files) + 1
        
        cap = cv2.VideoCapture(0)
        fps = 30
        
        print(f"\n{'='*60}")
        print(f"Recording {num_samples} samples for sign: {sign_name}")
        print(f"Duration: {duration_sec} seconds per sample")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            sample_idx = start_idx + i
            output_file = sign_dir / f"{sign_name}_{sample_idx:03d}.mp4"
            
            # Wait for user to get ready
            print(f"\nSample {i+1}/{num_samples}")
            print("Press SPACE when ready to record...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, f"Sign: {sign_name} | Sample: {i+1}/{num_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to start recording", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('ISL Data Recorder', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Record video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (640, 480))
            
            frames_to_record = duration_sec * fps
            for frame_idx in range(frames_to_record):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to standard size
                frame = cv2.resize(frame, (640, 480))
                
                # Add recording indicator
                remaining = (frames_to_record - frame_idx) / fps
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {remaining:.1f}s", 
                           (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"{sign_name}", 
                           (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(frame)
                cv2.imshow('ISL Data Recorder', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            out.release()
            print(f"✓ Saved: {output_file.name}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Recording complete! Saved {num_samples} samples to {sign_dir}")
        print(f"{'='*60}\n")
    
    def download_sample_data(self):
        """
        Provide instructions for downloading ISL datasets
        """
        instructions = """
╔═══════════════════════════════════════════════════════════════╗
║           ISL Dataset Download Instructions                    ║
╚═══════════════════════════════════════════════════════════════╝

Due to the large size of ISL datasets, please manually download from:

1. INCLUDE Dataset (Recommended)
   - URL: https://zenodo.org/record/4010759
   - Contains: 263 ISL signs with multiple samples
   - Download and extract to: data/ISL_dataset/

2. Kaggle ISL Datasets
   - Search: "Indian Sign Language Dataset"
   - Popular datasets:
     * ISL Hand Gesture Recognition
     * Indian Sign Language Dataset
   - Download and organize by sign name

3. Custom Recording
   - Use: python src/data_collection.py --record
   - Record your own samples using webcam

4. YouTube ISL Videos
   - Search: "Indian Sign Language tutorial"
   - Download videos and extract frames
   - Organize by sign name

After downloading:
- Organize videos/images by sign name in folders
- Run: python src/landmark_extraction.py
- Train model: python src/train.py

Target: 100+ samples per sign for 99% accuracy
"""
        print(instructions)
        
        # Save instructions
        with open(self.base_dir / 'DOWNLOAD_INSTRUCTIONS.txt', 'w') as f:
            f.write(instructions)
    
    def validate_dataset(self):
        """Check dataset completeness"""
        sign_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]
        
        stats = {
            'total_signs': len(sign_folders),
            'total_samples': 0,
            'signs_with_data': 0,
            'signs_needing_data': [],
            'samples_per_sign': {}
        }
        
        for sign_folder in sign_folders:
            videos = list(sign_folder.glob('*.mp4')) + list(sign_folder.glob('*.avi'))
            images = list(sign_folder.glob('*.jpg')) + list(sign_folder.glob('*.png'))
            
            num_samples = len(videos) + len(images)
            stats['total_samples'] += num_samples
            stats['samples_per_sign'][sign_folder.name] = num_samples
            
            if num_samples > 0:
                stats['signs_with_data'] += 1
            else:
                stats['signs_needing_data'].append(sign_folder.name)
        
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Total Signs: {stats['total_signs']}")
        print(f"Signs with Data: {stats['signs_with_data']}")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Average Samples per Sign: {stats['total_samples'] / max(stats['total_signs'], 1):.1f}")
        
        if stats['signs_needing_data']:
            print(f"\n⚠ Signs Needing Data ({len(stats['signs_needing_data'])}):")
            for sign in stats['signs_needing_data'][:10]:
                print(f"  - {sign}")
            if len(stats['signs_needing_data']) > 10:
                print(f"  ... and {len(stats['signs_needing_data']) - 10} more")
        
        print("\n" + "="*60)
        
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ISL Dataset Collection')
    parser.add_argument('--setup', action='store_true', 
                       help='Create dataset structure')
    parser.add_argument('--download', action='store_true',
                       help='Show download instructions')
    parser.add_argument('--record', action='store_true',
                       help='Record sign samples')
    parser.add_argument('--sign', type=str, help='Sign name to record')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to record')
    parser.add_argument('--duration', type=int, default=3,
                       help='Duration per sample in seconds')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset')
    
    args = parser.parse_args()
    
    collector = ISLDatasetCollector()
    
    if args.setup:
        collector.create_sample_dataset_structure()
    
    if args.download:
        collector.download_sample_data()
    
    if args.record:
        if not args.sign:
            print("Error: --sign argument required for recording")
        else:
            collector.record_sign_samples(args.sign, args.samples, args.duration)
    
    if args.validate:
        collector.validate_dataset()
    
    if not any([args.setup, args.download, args.record, args.validate]):
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  1. Setup: python src/data_collection.py --setup")
        print("  2. Download data (see instructions)")
        print("  3. Validate: python src/data_collection.py --validate")
