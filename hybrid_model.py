import tensorflow as tf
from tensorflow.keras import layers, models

class HybridModel:
    """Hybrid CNN-LSTM model for neuro-motor pattern recognition"""
    
    def __init__(self, cnn_features=256, lstm_units=128, dropout=0.4):
        self.cnn_features = cnn_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        
    def build_model(self, input_shape_cnn, input_shape_lstm, n_classes=3):
        """Build hybrid CNN-LSTM model"""
        
        # CNN branch for spatial features
        cnn_input = layers.Input(shape=input_shape_cnn, name='cnn_input')
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(cnn_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        cnn_output = layers.Dense(self.cnn_features, activation='relu')(x)
        
        # LSTM branch for temporal features
        lstm_input = layers.Input(shape=input_shape_lstm, name='lstm_input')
        
        y = layers.LSTM(self.lstm_units, return_sequences=True)(lstm_input)
        y = layers.Dropout(self.dropout)(y)
        y = layers.LSTM(self.lstm_units//2)(y)
        lstm_output = layers.Dense(64, activation='relu')(y)
        
        # Merge branches
        merged = layers.Concatenate()([cnn_output, lstm_output])
        
        # Dense layers for classification
        z = layers.Dense(128, activation='relu')(merged)
        z = layers.Dropout(self.dropout)(z)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dropout(self.dropout/2)(z)
        
        # Output layer
        output = layers.Dense(n_classes, activation='softmax', name='output')(z)
        
        # Create model
        self.model = models.Model(
            inputs=[cnn_input, lstm_input],
            outputs=output,
            name='Hybrid_CNN_LSTM'
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return self.model
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the hybrid model"""
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./datasets/models/hybrid_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_data[0],  # [cnn_input, lstm_input]
            train_data[1],  # labels
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, cnn_input, lstm_input):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be loaded or built before prediction")
        
        predictions = self.model.predict([cnn_input, lstm_input])
        return predictions
    
    def save(self, filepath):
        """Save model"""
        if self.model:
            self.model.save(filepath)
    
    def load(self, filepath):
        """Load model"""
        self.model = tf.keras.models.load_model(filepath)