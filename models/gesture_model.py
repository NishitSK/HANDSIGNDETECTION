"""
Advanced Deep Learning Model for ISL Recognition
LSTM/Transformer-based architecture for high accuracy gesture recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class ISLGestureModel:
    """Advanced gesture recognition model"""
    
    def __init__(self, config):
        """
        Initialize model with configuration
        
        Args:
            config: Configuration dictionary or object
        """
        self.config = config
        self.model = None
    
    def build_mlp_model(self, input_shape, num_classes):
        """
        Single-frame MLP classifier — no time convolution, no attention.

        Our pipeline already tiles/repeats one static photo across the whole
        `sequence_length` window (a held pose looks the same in every frame,
        see prepare_sequences), so a temporal model over that window has no
        real temporal signal to learn from — it's a heavier way to classify
        one frame. Built for weak/no-GPU phones: TCN's attention (Einsum) has
        no WASM kernel and takes ~5s/prediction on plain-JS CPU fallback; this
        MLP is a few Dense layers and runs in single-digit ms on CPU too, so
        classification isn't gated on WebGL being fast (or present) on the
        device. input_shape is (1, features) — same on-disk pipeline as every
        other architecture here, just with sequence_length=1 in config.

        Args:
            input_shape: (1, features)
            num_classes: Number of sign classes

        Returns:
            Compiled Keras model
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']

        hidden_units = model_config.get('hidden_units', 128)
        dropout = model_config.get('dropout', 0.3)
        learning_rate = model_config.get('learning_rate', 0.001)

        inputs = keras.Input(shape=input_shape, name='landmark_frame')
        x = layers.Flatten()(inputs)

        for i, units in enumerate([hidden_units, hidden_units // 2, hidden_units // 4]):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout if i == 0 else dropout * 0.5)(x)

        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='ISL_MLP_Model')

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_lstm_model(self, input_shape, num_classes):
        """
        Build LSTM-based model for sequence classification
        
        Args:
            input_shape: (sequence_length, features)
            num_classes: Number of sign classes
        
        Returns:
            Compiled Keras model
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']
        
        hidden_units = model_config.get('hidden_units', 256)
        num_layers = model_config.get('num_layers', 3)
        dropout = model_config.get('dropout', 0.3)
        learning_rate = model_config.get('learning_rate', 0.0001)
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='landmark_sequence')
        
        # Masking layer for variable length sequences
        x = layers.Masking(mask_value=0.0)(inputs)
        
        # Bidirectional LSTM layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            
            x = layers.Bidirectional(
                layers.LSTM(
                    hidden_units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=dropout * 0.5,
                    name=f'lstm_{i+1}'
                )
            )(x)
            
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        
        # Dense layers
        x = layers.Dense(hidden_units, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(hidden_units // 2, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ISL_LSTM_Model')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Metrics (Top-K disabled for sparse labels)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        return model
    
    def build_gru_model(self, input_shape, num_classes):
        """
        Build GRU-based model (faster alternative to LSTM)
        
        Args:
            input_shape: (sequence_length, features)
            num_classes: Number of sign classes
        
        Returns:
            Compiled Keras model
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']
        
        hidden_units = model_config.get('hidden_units', 256)
        num_layers = model_config.get('num_layers', 3)
        dropout = model_config.get('dropout', 0.3)
        learning_rate = model_config.get('learning_rate', 0.0001)
        
        inputs = keras.Input(shape=input_shape, name='landmark_sequence')
        x = layers.Masking(mask_value=0.0)(inputs)
        
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            
            x = layers.Bidirectional(
                layers.GRU(
                    hidden_units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=dropout * 0.5,
                    name=f'gru_{i+1}'
                )
            )(x)
            
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(hidden_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(hidden_units // 2, activation='relu')(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ISL_GRU_Model')
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Metrics (Top-K disabled for sparse labels)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        return model
    
    def build_tcn_model(self, input_shape, num_classes):
        """
        Build TCN (Temporal Convolutional Network) + Attention model
        FASTER training and HIGHER accuracy than LSTM
        
        Args:
            input_shape: (sequence_length, features)
            num_classes: Number of sign classes
        
        Returns:
            Compiled Keras model
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']
        
        hidden_units = model_config.get('hidden_units', 128)
        num_filters = hidden_units
        kernel_size = 3
        num_tcn_blocks = 2  # Reduced from 4 to prevent overfitting
        dropout = model_config.get('dropout', 0.2)
        learning_rate = model_config.get('learning_rate', 0.0005)
        
        inputs = keras.Input(shape=input_shape, name='landmark_sequence')
        
        # Initial projection
        x = layers.Conv1D(num_filters, 1, padding='same')(inputs)
        
        # TCN blocks with dilated convolutions
        for i in range(num_tcn_blocks):
            dilation_rate = 2 ** i
            
            # Residual block
            residual = x
            
            # Dilated causal convolution
            x = layers.Conv1D(
                num_filters, 
                kernel_size, 
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                name=f'tcn_conv1_{i}'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            
            x = layers.Conv1D(
                num_filters, 
                kernel_size, 
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                name=f'tcn_conv2_{i}'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            
            # Residual connection
            x = layers.Add()([x, residual])
        
        # Self-Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=num_filters // 8,
            name='attention'
        )(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(hidden_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(hidden_units // 2, activation='relu')(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ISL_TCN_Model')
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Metrics (Top-K disabled for sparse labels)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        return model
    
    def build_transformer_model(self, input_shape, num_classes):
        """
        Build Transformer-based model for sequence classification
        
        Args:
            input_shape: (sequence_length, features)
            num_classes: Number of sign classes
        
        Returns:
            Compiled Keras model
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']
        
        hidden_units = model_config.get('hidden_units', 256)
        num_heads = 8
        ff_dim = hidden_units * 2
        num_transformer_blocks = 4
        dropout = model_config.get('dropout', 0.3)
        learning_rate = model_config.get('learning_rate', 0.0001)
        
        inputs = keras.Input(shape=input_shape, name='landmark_sequence')
        
        # Positional encoding
        x = PositionalEncoding(hidden_units)(inputs)
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(hidden_units, num_heads, ff_dim, dropout)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(hidden_units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(hidden_units // 2, activation='relu')(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ISL_Transformer_Model')
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Metrics (Top-K disabled for sparse labels)
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        return model

    def build_seq2seq_transformer(self, input_shape, num_classes):
        """
        Build a lightweight Transformer encoder-decoder for sequence-to-sequence tasks.
        This implementation uses an encoder stack over the input landmark sequence and
        a compact decoder that produces a short token sequence (default target length = 1).
        """
        model_config = self.config['model'] if isinstance(self.config, dict) else self.config.config['model']

        d_model = model_config.get('d_model', 128)
        num_heads = model_config.get('num_heads', 4)
        ff_dim = model_config.get('ff_dim', d_model * 2)
        num_blocks = model_config.get('num_transformer_blocks', 2)
        dropout = model_config.get('dropout', 0.2)
        target_len = model_config.get('target_seq_length', 1)
        learning_rate = model_config.get('learning_rate', 0.0001)

        inputs = keras.Input(shape=input_shape, name='encoder_input')

        # Project to d_model and add positional encoding
        x = layers.Dense(d_model)(inputs)
        x = PositionalEncoding(d_model)(x)

        # Encoder blocks
        for _ in range(num_blocks):
            x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

        encoder_output = x

        # Simple decoder: start from pooled encoder state expanded to target length
        pooled = layers.GlobalAveragePooling1D()(encoder_output)
        y = layers.Dense(d_model * target_len, activation='relu')(pooled)
        y = layers.Reshape((target_len, d_model))(y)

        # Decoder blocks with cross-attention to encoder outputs
        for _ in range(num_blocks):
            # Self-attention on decoder
            sa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(y, y)
            y = layers.Add()([y, sa])
            y = layers.LayerNormalization(epsilon=1e-6)(y)

            # Cross-attention: decoder queries encoder
            ca = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(y, encoder_output)
            y = layers.Add()([y, ca])
            y = layers.LayerNormalization(epsilon=1e-6)(y)

            # Feed-forward
            ffn = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(d_model)])(y)
            y = layers.Add()([y, ffn])
            y = layers.LayerNormalization(epsilon=1e-6)(y)

        # Output tokens (time-distributed softmax)
        outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'), name='decoder_output')(y)

        model = Model(inputs=inputs, outputs=outputs, name='ISL_Seq2Seq_Transformer')

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def build(self, input_shape, num_classes):
        """
        Build model based on config
        
        Args:
            input_shape: (sequence_length, features)
            num_classes: Number of sign classes
        
        Returns:
            Compiled Keras model
        """
        # Get model type from config safely
        try:
            model_type = self.config['model'].get('type', 'LSTM').upper()
        except Exception:
            # Fallback
            model_type = 'LSTM'

        if model_type == 'MLP':
            self.model = self.build_mlp_model(input_shape, num_classes)
        elif model_type == 'LSTM':
            self.model = self.build_lstm_model(input_shape, num_classes)
        elif model_type == 'GRU':
            self.model = self.build_gru_model(input_shape, num_classes)
        elif model_type == 'TCN':
            self.model = self.build_tcn_model(input_shape, num_classes)
        elif model_type == 'TRANSFORMER':
            self.model = self.build_transformer_model(input_shape, num_classes)
        elif model_type == 'SEQ2SEQ':
            # For seq2seq, num_classes is the target vocabulary size
            self.model = self.build_seq2seq_transformer(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build() first.")


class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.projection = layers.Dense(d_model)
    
    def call(self, inputs):
        # Project inputs to d_model dimensions
        x = self.projection(inputs)
        
        # Add positional encoding
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        position_embeddings = self.get_position_encoding(length, self.d_model)
        
        return x + position_embeddings
    
    def get_position_encoding(self, length, d_model):
        """Calculate positional encoding"""
        positions = tf.range(start=0, limit=length, delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, 1)
        
        depths = tf.range(start=0, limit=d_model, delta=2, dtype=tf.float32)
        depths = depths / tf.cast(d_model, tf.float32)
        
        angle_rates = 1 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices, cos to odd
        sines = tf.sin(angle_rads)
        cosines = tf.cos(angle_rads)
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[:, :d_model]

        return tf.cast(pos_encoding, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class TransformerBlock(layers.Layer):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


def create_model(config, input_shape, num_classes):
    """
    Factory function to create model
    
    Args:
        config: Configuration dict or object
        input_shape: (sequence_length, features)
        num_classes: Number of sign classes
    
    Returns:
        Compiled Keras model
    """
    model_builder = ISLGestureModel(config)
    model = model_builder.build(input_shape, num_classes)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    return model


if __name__ == "__main__":
    # Test model creation
    from utils.config_loader import get_config
    
    config = get_config()
    
    # Example parameters
    sequence_length = config.get('model', 'sequence_length', default=30)
    input_features = config.get('model', 'input_features', default=63)
    num_classes = config.get('model', 'num_classes', default=100)
    
    input_shape = (sequence_length, input_features)
    
    print(f"Creating model with:")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    model = create_model(config, input_shape, num_classes)
