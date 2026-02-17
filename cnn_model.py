import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CNNModel:
    """CNN model for spatial pattern recognition in handwriting"""
    
    def __init__(self, input_shape=(224, 224, 1), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, architecture='custom'):
        """Build CNN model with specified architecture"""
        if architecture == 'resnet':
            self.model = self._build_resnet_like()
        elif architecture == 'vgg':
            self.model = self._build_vgg_like()
        else:  # custom
            self.model = self._build_custom_cnn()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def _build_custom_cnn(self):
        """Build custom CNN architecture for handwriting analysis"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block (deeper features)
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_resnet_like(self):
        """Build ResNet-like architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 128, downsample=True)
        x = self._residual_block(x, 256, downsample=True)
        x = self._residual_block(x, 512, downsample=True)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _residual_block(self, x, filters, downsample=False):
        """Residual block for ResNet-like architecture"""
        identity = x
        
        if downsample:
            identity = layers.Conv2D(filters, (1, 1), strides=2, padding='same')(identity)
            identity = layers.BatchNormalization()(identity)
        
        # First convolution
        strides = 2 if downsample else 1
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add shortcut
        x = layers.Add()([x, identity])
        x = layers.Activation('relu')(x)
        
        return x
    
    def _build_vgg_like(self):
        """Build VGG-like architecture"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=2),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=2),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=2),
            
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=2),
            
            # Block 5
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=2),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self, train_data, val_data, epochs=50, batch_size=32, callbacks=None):
        """Train the CNN model"""
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='./datasets/models/cnn_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs/cnn',
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
            verbose=1
        )
        
        return self.history
    
    def predict(self, images):
        """Make predictions on new images"""
        if self.model is None:
            raise ValueError("Model must be loaded or built before prediction")
        
        # Preprocess images if needed
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
        
        # Ensure correct shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        predictions = self.model.predict(images, verbose=0)
        return predictions
    
    def predict_single(self, image):
        """Predict single image"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.predict(image)
        return predictions[0]
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model must be loaded or built before evaluation")
        
        results = self.model.evaluate(test_data[0], test_data[1], verbose=0)
        
        metrics = {}
        for i, metric in enumerate(self.model.metrics_names):
            metrics[metric] = results[i]
        
        return metrics
    
    def get_feature_maps(self, image, layer_name=None):
        """Extract feature maps from specific layer"""
        if self.model is None:
            raise ValueError("Model must be loaded or built")
        
        # Create feature extractor model
        if layer_name:
            layer_output = self.model.get_layer(layer_name).output
        else:
            # Use last convolutional layer
            conv_layers = [layer for layer in self.model.layers 
                          if isinstance(layer, layers.Conv2D)]
            if conv_layers:
                layer_output = conv_layers[-1].output
            else:
                raise ValueError("No convolutional layers found in model")
        
        feature_extractor = models.Model(inputs=self.model.input, 
                                        outputs=layer_output)
        
        # Prepare image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Extract features
        features = feature_extractor.predict(image, verbose=0)
        return features
    
    def visualize_feature_maps(self, image, num_maps=16):
        """Visualize feature maps"""
        import matplotlib.pyplot as plt
        
        # Get feature maps from last conv layer
        feature_maps = self.get_feature_maps(image)
        
        # Plot feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_maps, feature_maps.shape[-1])):
            ax = axes[i]
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'Feature Map {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath):
        """Save model to file"""
        if self.model:
            self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from file"""
        self.model = tf.keras.models.load_model(filepath)
    
    def get_class_activation_map(self, image, class_idx=None):
        """Generate Class Activation Map (CAM)"""
        if self.model is None:
            raise ValueError("Model must be loaded or built")
        
        # Get last convolutional layer
        conv_layers = [layer for layer in self.model.layers 
                      if isinstance(layer, layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No convolutional layers found")
        
        last_conv_layer = conv_layers[-1]
        
        # Create model that outputs last conv layer and predictions
        cam_model = models.Model(
            inputs=self.model.input,
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Prepare image
        if len(image.shape) == 2:
            image_input = np.expand_dims(image, axis=0)
            image_input = np.expand_dims(image_input, axis=-1)
        elif len(image.shape) == 3:
            image_input = np.expand_dims(image, axis=0)
        
        # Get conv output and predictions
        conv_outputs, predictions = cam_model.predict(image_input, verbose=0)
        conv_output = conv_outputs[0]
        
        # If no class specified, use predicted class
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        
        # Get weights for the class
        class_weights = self.model.layers[-1].get_weights()[0]
        
        # Create CAM
        cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)
        
        for i, w in enumerate(class_weights[:, class_idx]):
            cam += w * conv_output[:, :, i]
        
        # Resize CAM to original image size
        import cv2
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam, predictions[0]