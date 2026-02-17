import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    """LSTM model for temporal pattern analysis in handwriting"""
    
    def __init__(self, input_shape=(100, 8), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self, architecture='bidirectional'):
        """Build LSTM model with specified architecture"""
        if architecture == 'bidirectional':
            self.model = self._build_bidirectional_lstm()
        elif architecture == 'stacked':
            self.model = self._build_stacked_lstm()
        elif architecture == 'attention':
            self.model = self._build_attention_lstm()
        else:  # simple
            self.model = self._build_simple_lstm()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def _build_simple_lstm(self):
        """Build simple LSTM model"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_bidirectional_lstm(self):
        """Build bidirectional LSTM model"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3)),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_stacked_lstm(self):
        """Build stacked LSTM model"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_attention_lstm(self):
        """Build LSTM with attention mechanism"""
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layer
        lstm_out = layers.LSTM(128, return_sequences=True, dropout=0.3)(inputs)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        attention_out = layers.Multiply()([lstm_out, attention])
        attention_out = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_out)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(attention_out)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def prepare_sequence_data(self, stroke_sequences, labels=None, sequence_length=100):
        """Prepare sequence data for LSTM"""
        sequences = []
        processed_labels = []
        
        for i, stroke_seq in enumerate(stroke_sequences):
            # Convert stroke sequence to feature matrix
            features = self._extract_sequence_features(stroke_seq)
            
            # Pad or truncate to fixed length
            if len(features) > sequence_length:
                # Truncate
                features = features[:sequence_length]
            else:
                # Pad with zeros
                padding = np.zeros((sequence_length - len(features), features.shape[1]))
                features = np.vstack([features, padding])
            
            sequences.append(features)
            
            if labels is not None:
                processed_labels.append(labels[i])
        
        sequences = np.array(sequences)
        
        if labels is not None:
            return sequences, np.array(processed_labels)
        else:
            return sequences
    
    def _extract_sequence_features(self, stroke_sequence):
        """Extract features from stroke sequence for LSTM"""
        if len(stroke_sequence) == 0:
            return np.zeros((1, 8))
        
        features = []
        
        for point in stroke_sequence:
            if isinstance(point, dict):
                # Process dictionary format
                feature_vector = [
                    point.get('x', 0),
                    point.get('y', 0),
                    point.get('pressure', 0.5),
                    point.get('vx', 0) if 'vx' in point else 0,
                    point.get('vy', 0) if 'vy' in point else 0,
                    point.get('speed', 0) if 'speed' in point else 0,
                    point.get('acceleration', 0) if 'acceleration' in point else 0,
                    point.get('curvature', 0) if 'curvature' in point else 0
                ]
            else:
                # Assume numpy array format
                if len(point) >= 8:
                    feature_vector = point[:8]
                else:
                    feature_vector = list(point) + [0] * (8 - len(point))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def normalize_sequences(self, sequences):
        """Normalize sequence data"""
        original_shape = sequences.shape
        flattened = sequences.reshape(-1, sequences.shape[-1])
        
        # Fit scaler if not fitted
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(flattened)
        
        normalized = self.scaler.transform(flattened)
        normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def train(self, train_data, val_data, epochs=100, batch_size=32, callbacks=None):
        """Train the LSTM model"""
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='./datasets/models/lstm_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs/lstm',
                    histogram_freq=1
                )
            ]
        
        # Train the model
        self.history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history
    
    def predict(self, sequences):
        """Make predictions on new sequences"""
        if self.model is None:
            raise ValueError("Model must be loaded or built before prediction")
        
        # Normalize sequences
        sequences_normalized = self.normalize_sequences(sequences)
        
        predictions = self.model.predict(sequences_normalized, verbose=0)
        return predictions
    
    def predict_sequence(self, stroke_sequence):
        """Predict single stroke sequence"""
        # Prepare sequence
        sequences = self.prepare_sequence_data([stroke_sequence])
        predictions = self.predict(sequences)
        return predictions[0]
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model must be loaded or built before evaluation")
        
        # Normalize test data
        X_test_normalized = self.normalize_sequences(test_data[0])
        
        results = self.model.evaluate(X_test_normalized, test_data[1], verbose=0)
        
        metrics = {}
        for i, metric in enumerate(self.model.metrics_names):
            metrics[metric] = results[i]
        
        return metrics
    
    def get_sequence_embeddings(self, sequences):
        """Get sequence embeddings from LSTM layer"""
        if self.model is None:
            raise ValueError("Model must be loaded or built")
        
        # Create embedding model (output from last LSTM layer)
        lstm_layers = [layer for layer in self.model.layers 
                      if isinstance(layer, (layers.LSTM, layers.Bidirectional))]
        if not lstm_layers:
            raise ValueError("No LSTM layers found in model")
        
        # Get the last LSTM layer
        last_lstm_layer = lstm_layers[-1]
        
        # Create embedding model
        embedding_model = models.Model(
            inputs=self.model.input,
            outputs=last_lstm_layer.output
        )
        
        # Normalize sequences
        sequences_normalized = self.normalize_sequences(sequences)
        
        # Get embeddings
        embeddings = embedding_model.predict(sequences_normalized, verbose=0)
        return embeddings
    
    def visualize_attention(self, sequence, layer_name='attention'):
        """Visualize attention weights for a sequence"""
        if self.model is None:
            raise ValueError("Model must be loaded or built")
        
        # Check if model has attention
        attention_layers = [layer for layer in self.model.layers 
                           if 'attention' in layer.name.lower()]
        if not attention_layers:
            raise ValueError("Model does not have attention mechanism")
        
        # Create model that outputs attention weights
        attention_layer = attention_layers[0]
        attention_model = models.Model(
            inputs=self.model.input,
            outputs=attention_layer.output
        )
        
        # Prepare sequence
        sequences = self.prepare_sequence_data([sequence])
        sequences_normalized = self.normalize_sequences(sequences)
        
        # Get attention weights
        attention_weights = attention_model.predict(sequences_normalized, verbose=0)
        
        # Flatten attention weights
        attention_weights = attention_weights[0].flatten()
        
        # Truncate to sequence length
        actual_length = min(len(sequence), len(attention_weights))
        attention_weights = attention_weights[:actual_length]
        
        return attention_weights
    
    def analyze_temporal_patterns(self, sequences):
        """Analyze temporal patterns in sequences"""
        if self.model is None:
            raise ValueError("Model must be loaded or built")
        
        # Get predictions
        predictions = self.predict(sequences)
        
        # Get sequence embeddings
        embeddings = self.get_sequence_embeddings(sequences)
        
        # Analyze temporal patterns
        patterns = []
        
        for i, (pred, emb) in enumerate(zip(predictions, embeddings)):
            pattern = {
                'prediction': np.argmax(pred),
                'confidence': float(np.max(pred)),
                'embedding': emb.tolist(),
                'temporal_features': self._extract_temporal_features(sequences[i])
            }
            patterns.append(pattern)
        
        return patterns
    
    def _extract_temporal_features(self, sequence):
        """Extract temporal features from sequence"""
        if len(sequence) == 0:
            return {}
        
        # Calculate various temporal statistics
        sequence_flat = sequence.reshape(-1, sequence.shape[-1])
        
        features = {
            'sequence_length': sequence.shape[0],
            'mean_values': np.mean(sequence_flat, axis=0).tolist(),
            'std_values': np.std(sequence_flat, axis=0).tolist(),
            'temporal_variability': np.mean(np.std(sequence, axis=0)),
            'autocorrelation_lag1': self._calculate_autocorrelation(sequence_flat[:, 0], lag=1),
            'dominant_frequency': self._calculate_dominant_frequency(sequence_flat[:, 0])
        }
        
        return features
    
    def _calculate_autocorrelation(self, signal, lag=1):
        """Calculate autocorrelation at given lag"""
        if len(signal) <= lag:
            return 0
        
        mean = np.mean(signal)
        var = np.var(signal)
        
        if var == 0:
            return 0
        
        autocorr = np.mean((signal[lag:] - mean) * (signal[:-lag] - mean)) / var
        return float(autocorr)
    
    def _calculate_dominant_frequency(self, signal):
        """Calculate dominant frequency using FFT"""
        if len(signal) < 10:
            return 0
        
        # Remove DC component
        signal_detrended = signal - np.mean(signal)
        
        # Calculate FFT
        fft_values = np.fft.fft(signal_detrended)
        frequencies = np.fft.fftfreq(len(signal_detrended))
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_values)
        
        # Find dominant frequency (skip DC)
        dominant_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_freq = abs(frequencies[dominant_idx])
        
        return float(dominant_freq)
    
    def save(self, filepath):
        """Save model to file"""
        if self.model:
            self.model.save(filepath)
            # Also save scaler
            import joblib
            joblib.dump(self.scaler, filepath.replace('.h5', '_scaler.pkl'))
    
    def load(self, filepath):
        """Load model from file"""
        self.model = tf.keras.models.load_model(filepath)
        # Try to load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
        except:
            print(f"Warning: Could not load scaler from {scaler_path}")