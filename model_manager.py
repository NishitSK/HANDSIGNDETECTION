"""
Unified Model Manager for ISL System
Manages multiple trained models (alphabets, numbers, phrases, gestures)
Automatically selects and loads the best model for each category
"""
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np


class ModelManager:
    """Manages multiple ISL models"""
    
    def __init__(self, checkpoints_dir='models/checkpoints'):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.models = {}
        self.metadata = {}
        
        # Model categories
        self.categories = {
            'alphabets': None,
            'numbers': None,
            'phrases': None,
            'gestures': None
        }
        
        self.scan_models()
    
    def scan_models(self):
        """Scan checkpoint directory for trained models"""
        print("\n" + "="*60)
        print("SCANNING TRAINED MODELS")
        print("="*60)
        
        if not self.checkpoints_dir.exists():
            print("⚠ No checkpoints directory found")
            return
        
        model_files = sorted(self.checkpoints_dir.glob("*.h5"))
        
        for model_path in model_files:
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_name = metadata.get('model_name', 'Unknown')
                num_classes = metadata.get('num_classes', 0)
                try:
                    accuracy = float(model_path.stem.split('_')[-1])
                except ValueError:
                    accuracy = metadata.get('accuracy', 0.0)
                
                # Categorize model
                category = self._categorize_model(metadata)
                
                if category:
                    # Keep best model for each category
                    if category not in self.categories or self.categories[category] is None:
                        self.categories[category] = {
                            'path': model_path,
                            'metadata': metadata,
                            'accuracy': accuracy
                        }
                    elif accuracy > self.categories[category]['accuracy']:
                        self.categories[category] = {
                            'path': model_path,
                            'metadata': metadata,
                            'accuracy': accuracy
                        }
                
                print(f"✓ Found: {model_path.name}")
                print(f"  Category: {category}")
                print(f"  Classes: {num_classes}, Accuracy: {accuracy:.1%}")
        
        print("\n" + "="*60)
        print("BEST MODELS BY CATEGORY")
        print("="*60)
        
        for category, info in self.categories.items():
            if info:
                print(f"\n{category.upper()}:")
                print(f"  Model: {info['path'].name}")
                print(f"  Accuracy: {info['accuracy']:.2%}")
                print(f"  Classes: {info['metadata']['num_classes']}")
            else:
                print(f"\n{category.upper()}: Not trained yet")
        
        print("\n" + "="*60)
    
    def _categorize_model(self, metadata):
        """Determine model category from metadata"""
        model_name = metadata.get('model_name', '').lower()
        class_names = metadata.get('class_names', [])
        num_classes = metadata.get('num_classes', 0)

        if not class_names:
            return 'gestures'

        # Check class names to determine category
        if all(c.isalpha() and len(c) == 1 for c in class_names[:5]):
            return 'alphabets'
        elif all(c.isdigit() for c in class_names[:5]):
            return 'numbers'
        elif 'letter' in model_name or num_classes == 26:
            return 'alphabets'
        elif 'number' in model_name or (num_classes <= 10 and num_classes >= 9):
            return 'numbers'
        elif 'phrase' in model_name or len(class_names[0].split()) > 1:
            return 'phrases'
        else:
            return 'gestures'
    
    def load_model(self, category='alphabets'):
        """Load the best model for a category"""
        if category not in self.categories or self.categories[category] is None:
            raise ValueError(f"No trained model found for category: {category}")
        
        info = self.categories[category]
        
        if category not in self.models:
            print(f"\nLoading {category} model...")
            self.models[category] = keras.models.load_model(info['path'])
            self.metadata[category] = info['metadata']
            print(f"✓ Loaded: {info['path'].name}")
        
        return self.models[category], self.metadata[category]
    
    def get_best_model_path(self, category='alphabets'):
        """Get path to best model for category"""
        if category in self.categories and self.categories[category]:
            return str(self.categories[category]['path'])
        return None
    
    def update_config(self, config_path='config.yaml', category='alphabets'):
        """Update config.yaml with best model path"""
        import yaml
        
        model_path = self.get_best_model_path(category)
        if not model_path:
            print(f"⚠ No model found for {category}")
            return
        
        # Read config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model path
        config['paths']['model_path'] = model_path
        
        # Update model settings
        info = self.categories[category]
        config['model']['num_classes'] = info['metadata']['num_classes']
        config['model']['input_features'] = info['metadata']['input_shape'][1]
        config['model']['sequence_length'] = info['metadata']['input_shape'][0]
        config['model']['name'] = info['metadata']['model_name']
        
        # Write back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Updated config.yaml with {category} model")
        print(f"  Path: {model_path}")


if __name__ == '__main__':
    # Test model manager
    manager = ModelManager()
    
    # Update config to use alphabet model
    manager.update_config(category='alphabets')
    
    print("\n✓ Model manager ready!")
