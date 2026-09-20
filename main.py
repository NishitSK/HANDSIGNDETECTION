"""
Main Entry Point for ISL Translation System
"""

import sys
import os
from pathlib import Path


def print_banner():
    """Print application banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║        American Sign Language Translation System               ║
    ║        Advanced Real-time Detection & Voice Conversion         ║
    ║                                                                ║
    ║        Version 1.0 - Professional Edition                      ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'mediapipe', 'tensorflow', 'transformers',
        'torch', 'pyttsx3', 'PyQt5', 'numpy', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\n❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def check_model():
    """Check if trained model exists"""
    from utils.config_loader import get_config
    
    config = get_config('config.yaml')
    model_path = Path(config['paths']['model_path'])
    
    if not model_path.exists():
        print(f"\n⚠️  Trained model not found: {model_path}")
        print("\n📋 Available options:")
        print("   1. Train alphabets:    Edit train.py (TRAIN_TYPE='letter') and run")
        print("   2. Train numbers:      Edit train.py (TRAIN_TYPE='number') and run")
        print("   3. Train phrases:      Edit train.py (TRAIN_TYPE='phrase') and run")
        print("\nOr use model_manager.py to select a different trained model.")
        return False
    
    return True


def show_menu():
    """Show application menu"""
    print("\n" + "="*60)
    print("MAIN MENU")
    print("="*60)
    print("\n1. 🎨 Launch GUI Application (Recommended)")
    print("2. � Collect Training Data (ISL_DATASETS_2)")
    print("3. 🚀 Train Model with Collected Data")
    print("4. 📹 Real-time Detection (Command Line)")
    print("5. 📊 Setup Dataset")
    print("6. 🔧 Test Components")
    print("7. ❌ Exit")
    print("\n" + "="*60)
    
    choice = input("\nSelect option (1-7): ").strip()
    return choice


def launch_gui():
    """Launch GUI application"""
    print("\n🚀 Launching GUI application...")
    from src.gui_app import main as gui_main
    gui_main()


def launch_cli():
    """Launch command-line inference"""
    print("\n🚀 Launching real-time detection...")
    from src.inference import ISLInference
    from utils.config_loader import get_config
    
    config = get_config('config.yaml')
    model_path = config['paths']['model_path']
    
    print(f"Loading model: {model_path}")
    inference = ISLInference(model_path)
    inference.run_camera(0)


def train_model():
    """Train model with collected data"""
    print("\n🎓 Training model with collected data...")
    print("📁 Looking for data in: data/ISL_DATASETS_2/")
    
    from pathlib import Path
    import subprocess
    
    data_dir = Path('data/ISL_DATASETS_2')
    
    if not data_dir.exists():
        print("\n❌ No collected data found!")
        print("\n💡 First collect training data:")
        print("   1. Select option 2 from main menu")
        print("   2. Press 'T' to enter class name")
        print("   3. Press SPACE to start 3-second timer for photo")
        print("   4. Collect 80+ photos per class")
        print("   5. Return here and select option 3 to train")
        return
    
    # Count classes and samples
    classes = []
    total_photos = 0
    total_videos = 0
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            photo_count = len(list(class_dir.glob('*.jpg')))
            video_count = len(list(class_dir.glob('*.avi')))
            
            if photo_count > 0 or video_count > 0:
                classes.append(class_dir.name)
                total_photos += photo_count
                total_videos += video_count
    
    if len(classes) == 0:
        print("\n❌ No data found in ISL_DATASETS_2!")
        print("   Collect data first (option 2)")
        return
    
    print(f"\n📊 Found training data:")
    print(f"   Classes: {len(classes)}")
    print(f"   Total Photos: {total_photos}")
    print(f"   Total Videos: {total_videos}")
    print(f"\n   Classes: {', '.join(classes[:10])}")
    if len(classes) > 10:
        print(f"   ...and {len(classes) - 10} more")
    
    confirm = input("\n▶️  Start training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Training cancelled")
        return
    
    # Update train.py configuration
    try:
        train_path = Path('train.py')
        with open(train_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to replace only the CLASSES in the custom section
        import re
        
        # Pattern to find the custom section and replace CLASSES
        pattern = r"(elif TRAIN_TYPE == \"custom\":.*?CLASSES = )(\[.*?\])"
        replacement = rf"\1{classes}"
        
        # Replace CLASSES in custom section
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Ensure TRAIN_TYPE is set to custom
        content = re.sub(r'TRAIN_TYPE = "[^"]*"', 'TRAIN_TYPE = "custom"', content)
        
        # Update DATA_PATH if it exists
        if 'DATA_PATH =' in content:
            content = re.sub(r'DATA_PATH = "[^"]*"', 'DATA_PATH = "data/ISL_DATASETS_2"', content)
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ train.py configured")
        print("🚀 Starting training...\n")
        
        # Run training
        subprocess.run(['python', 'train.py'])
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")


def collect_data():
    """Launch modern GUI data collection tool"""
    print("\n📸 Launching professional data collection GUI...")
    print("📁 Data will be saved to: data/ISL_DATASETS_2/\n")
    
    import subprocess
    subprocess.run(['python', 'gui_data_collector.py'])


def setup_dataset():
    """Setup dataset"""
    print("\n📊 Setting up dataset structure...")
    from src.data_collection import ISLDatasetCollector
    
    collector = ISLDatasetCollector()
    collector.create_sample_dataset_structure()
    collector.download_sample_data()
    
    print("\n✓ Dataset setup complete!")
    print("\nNext steps:")
    print("1. Download ISL videos/images (see instructions)")
    print("2. Populate data/ISL_dataset/ folders")
    print("3. Run: python src/landmark_extraction.py")
    print("4. Run: python src/train.py")


def test_components():
    """Test individual components"""
    print("\n🔧 Testing components...\n")
    
    tests = {
        '1': ('Grammar Correction', 'src.grammar_correction'),
        '2': ('Text-to-Speech', 'src.tts_engine'),
        '3': ('Landmark Extraction', 'src.landmark_extraction'),
        '4': ('Model Architecture', 'models.gesture_model')
    }
    
    print("Select component to test:")
    for key, (name, _) in tests.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice in tests:
        name, module = tests[choice]
        print(f"\n▶️  Testing {name}...")
        
        try:
            # Import and run module
            import importlib
            mod = importlib.import_module(module)
            # Module's __main__ section will run
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    else:
        print("Invalid choice")


def main():
    """Main application entry point"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check for trained model
    model_exists = check_model()
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            if model_exists:
                launch_gui()
            else:
                print("\n❌ Cannot launch GUI: Model not trained yet")
                print("💡 Collect data (option 2) and train (option 3) first")
        
        elif choice == '2':
            collect_data()
        
        elif choice == '3':
            train_model()
            model_exists = True  # Model should exist now
        
        elif choice == '4':
            if model_exists:
                launch_cli()
            else:
                print("\n❌ Cannot run inference: Model not trained yet")
                print("💡 Collect data (option 2) and train (option 3) first")
        
        elif choice == '5':
            setup_dataset()
        
        elif choice == '6':
            test_components()
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("\n❌ Invalid option. Please choose 1-7.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
