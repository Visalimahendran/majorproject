import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class StressClassifier:
    """Classify stress levels from handwriting features"""
    
    def __init__(self, model_type='random_forest', n_classes=3):
        self.model_type = model_type
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        self.class_labels = ['normal', 'mild_stress', 'severe_stress']
        self.feature_importance = None
        
    def build_model(self, params=None):
        """Build and configure the classifier model"""
        if params is None:
            params = self._get_default_params()
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**params)
        elif self.model_type == 'svm':
            self.model = SVC(**params, probability=True)
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def _get_default_params(self):
        """Get default parameters for each model type"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        elif self.model_type == 'svm':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            }
    
    def prepare_features(self, features_dict, feature_names=None):
        """Prepare features for classification"""
        if feature_names is None:
            # Use all available features
            feature_vector = []
            feature_names = []
            
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                    feature_names.append(key)
                elif isinstance(value, dict):
                    # Flatten nested dictionaries
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            feature_vector.append(sub_value)
                            feature_names.append(f"{key}_{sub_key}")
        else:
            # Use specified feature names
            feature_vector = []
            for name in feature_names:
                # Handle nested feature access
                if '_' in name:
                    main_key, sub_key = name.split('_', 1)
                    if main_key in features_dict and sub_key in features_dict[main_key]:
                        feature_vector.append(features_dict[main_key][sub_key])
                    else:
                        feature_vector.append(0.0)
                else:
                    feature_vector.append(features_dict.get(name, 0.0))
        
        return np.array(feature_vector).reshape(1, -1), feature_names
    
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=False):
        """Train the classifier"""
        if self.model is None:
            self.build_model()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize:
            # Perform hyperparameter optimization
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            self.build_model(best_params)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            return val_score
        
        return None
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using grid search"""
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': [0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        elif self.model_type == 'neural_network':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        else:
            return self._get_default_params()
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_
    
    def predict(self, features_dict, return_proba=False):
        """Make prediction on new features"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X, feature_names = self.prepare_features(features_dict)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            # Return probability distribution
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Create probability dictionary
            prob_dict = {}
            for i, label in enumerate(self.class_labels[:self.n_classes]):
                prob_dict[label] = float(probabilities[i])
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_labels[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'probabilities': prob_dict,
                'confidence': float(np.max(probabilities))
            }
        else:
            # Return class prediction only
            prediction = self.model.predict(X_scaled)[0]
            return self.class_labels[prediction]
    
    def predict_batch(self, features_list):
        """Predict multiple feature sets"""
        predictions = []
        
        for features in features_list:
            pred = self.predict(features, return_proba=True)
            predictions.append(pred)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC for each class
        from sklearn.metrics import roc_auc_score
        try:
            if self.n_classes == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            self.build_model()
        
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'fold_scores': scores.tolist()
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        if self.feature_importance is None:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            else:
                return None
        
        # Sort features by importance
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_dict = {}
        for i, (name, importance) in enumerate(zip(feature_names, self.feature_importance)):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_decision_explanation(self, features_dict, top_features=10):
        """Explain model decision for a prediction"""
        if self.model is None:
            raise ValueError("Model must be trained")
        
        # Get prediction
        prediction_result = self.predict(features_dict, return_proba=True)
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance is None:
            return {
                'prediction': prediction_result,
                'explanation': 'Feature importance not available for this model type.'
            }
        
        # Prepare features for contribution analysis
        X, feature_names = self.prepare_features(features_dict)
        X_scaled = self.scaler.transform(X)[0]
        
        # Calculate feature contributions (simplified)
        contributions = {}
        for i, (name, importance) in enumerate(feature_importance.items()):
            if i < len(X_scaled):
                # Contribution = importance * normalized feature value
                contributions[name] = {
                    'importance': importance,
                    'value': float(X_scaled[i]) if i < len(X_scaled) else 0.0,
                    'contribution': importance * (X_scaled[i] if i < len(X_scaled) else 0.0)
                }
        
        # Sort contributions by absolute value
        sorted_contributions = dict(sorted(contributions.items(), 
                                          key=lambda x: abs(x[1]['contribution']), 
                                          reverse=True))
        
        # Get top contributing features
        top_contributors = {}
        for i, (name, data) in enumerate(sorted_contributions.items()):
            if i >= top_features:
                break
            top_contributors[name] = data
        
        explanation = {
            'prediction': prediction_result,
            'top_contributing_features': top_contributors,
            'decision_factors': self._generate_decision_factors(top_contributors, 
                                                              prediction_result['predicted_class'])
        }
        
        return explanation
    
    def _generate_decision_factors(self, top_features, predicted_class):
        """Generate human-readable decision factors"""
        factors = []
        
        for feature_name, data in top_features.items():
            contribution = data['contribution']
            value = data['value']
            
            # Interpret based on feature name
            if 'tremor' in feature_name.lower():
                if contribution > 0:
                    factors.append(f"Elevated tremor ({value:.2f}) contributed to {predicted_class} classification")
                else:
                    factors.append(f"Reduced tremor ({value:.2f}) influenced classification")
            
            elif 'hesitation' in feature_name.lower():
                if contribution > 0:
                    factors.append(f"Increased hesitation frequency ({value:.2f}) was a factor")
                else:
                    factors.append(f"Lower hesitation rate ({value:.2f}) affected decision")
            
            elif 'pressure' in feature_name.lower():
                if contribution > 0:
                    factors.append(f"Higher pressure variability ({value:.2f}) contributed")
                else:
                    factors.append(f"Consistent pressure ({value:.2f}) was noted")
            
            elif 'velocity' in feature_name.lower() or 'speed' in feature_name.lower():
                if contribution > 0:
                    factors.append(f"Velocity irregularity ({value:.2f}) was significant")
                else:
                    factors.append(f"Smooth velocity ({value:.2f}) influenced classification")
            
            elif 'irregularity' in feature_name.lower():
                if contribution > 0:
                    factors.append(f"Stroke irregularity ({value:.2f}) was a key factor")
                else:
                    factors.append(f"Regular stroke patterns ({value:.2f}) affected decision")
        
        return factors
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'n_classes': self.n_classes,
            'class_labels': self.class_labels,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.n_classes = model_data['n_classes']
        self.class_labels = model_data['class_labels']
        self.feature_importance = model_data['feature_importance']
    
    def create_feature_report(self, features_dict):
        """Create detailed feature analysis report"""
        report = {
            'basic_statistics': {},
            'feature_categories': {},
            'anomalies': []
        }
        
        # Categorize features
        categories = {
            'tremor': ['tremor_intensity', 'tremor_ratio', 'tremor_frequency'],
            'hesitation': ['hesitation_ratio', 'pause_count', 'hesitation_time'],
            'pressure': ['pressure_mean', 'pressure_std', 'pressure_cv', 'pressure_spikes'],
            'velocity': ['velocity_mean', 'velocity_std', 'velocity_cv', 'velocity_spikes'],
            'irregularity': ['irregularity_index', 'jerk_cost', 'fractal_dimension'],
            'rhythm': ['rhythm_regularity', 'pause_expon_scale', 'spectral_entropy']
        }
        
        # Calculate basic statistics
        numeric_values = []
        for key, value in features_dict.items():
            if isinstance(value, (int, float)):
                numeric_values.append(value)
                report['basic_statistics'][key] = float(value)
        
        if numeric_values:
            report['basic_statistics']['summary'] = {
                'mean': float(np.mean(numeric_values)),
                'std': float(np.std(numeric_values)),
                'min': float(np.min(numeric_values)),
                'max': float(np.max(numeric_values)),
                'range': float(np.max(numeric_values) - np.min(numeric_values))
            }
        
        # Categorize features
        for category, feature_list in categories.items():
            category_values = []
            for feature in feature_list:
                if feature in features_dict:
                    category_values.append(features_dict[feature])
            
            if category_values:
                report['feature_categories'][category] = {
                    'mean': float(np.mean(category_values)),
                    'std': float(np.std(category_values)),
                    'count': len(category_values)
                }
        
        # Detect anomalies
        anomaly_thresholds = {
            'tremor_intensity': 0.5,
            'hesitation_ratio': 0.2,
            'pressure_cv': 0.3,
            'irregularity_index': 0.6,
            'velocity_cv': 0.4
        }
        
        for feature, threshold in anomaly_thresholds.items():
            if feature in features_dict and features_dict[feature] > threshold:
                report['anomalies'].append({
                    'feature': feature,
                    'value': float(features_dict[feature]),
                    'threshold': float(threshold),
                    'severity': float(min(1.0, features_dict[feature] / threshold))
                })
        
        return report