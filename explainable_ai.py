import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

class XAI:
    """Explainable AI module for model interpretability"""
    
    def __init__(self):
        self.shap_explainer = None
        self.feature_names = None
        
    def explain_model(self, model, X_train, X_test, feature_names=None):
        """Generate comprehensive model explanations"""
        explanations = {}
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # 1. Feature Importance
        explanations['feature_importance'] = self._calculate_feature_importance(model, X_test)
        
        # 2. SHAP Values
        explanations['shap_values'] = self._calculate_shap_values(model, X_train, X_test)
        
        # 3. Partial Dependence
        explanations['partial_dependence'] = self._calculate_partial_dependence(model, X_train)
        
        # 4. Decision Boundaries
        explanations['decision_boundaries'] = self._analyze_decision_boundaries(model, X_train)
        
        # 5. Model Confidence
        explanations['confidence_analysis'] = self._analyze_model_confidence(model, X_test)
        
        return explanations
    
    def _calculate_feature_importance(self, model, X_test, n_repeats=10):
        """Calculate permutation feature importance"""
        try:
            # Calculate permutation importance
            result = permutation_importance(
                model, X_test, 
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            importance = {
                'importances_mean': result.importances_mean.tolist(),
                'importances_std': result.importances_std.tolist(),
                'feature_names': self.feature_names
            }
            
            # Sort features by importance
            sorted_idx = np.argsort(result.importances_mean)[::-1]
            sorted_features = []
            
            for i in sorted_idx:
                sorted_features.append({
                    'feature': self.feature_names[i],
                    'importance_mean': float(result.importances_mean[i]),
                    'importance_std': float(result.importances_std[i])
                })
            
            importance['sorted_features'] = sorted_features
            
            # Create visualization data
            importance['visualization'] = {
                'x': [sf['feature'] for sf in sorted_features[:10]],
                'y': [sf['importance_mean'] for sf in sorted_features[:10]],
                'error': [sf['importance_std'] for sf in sorted_features[:10]]
            }
            
            return importance
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return {'error': str(e)}
    
    def _calculate_shap_values(self, model, X_train, X_test, n_samples=100):
        """Calculate SHAP values for model explanation"""
        try:
            # Sample data for SHAP calculation (for efficiency)
            if len(X_train) > n_samples:
                X_train_sample = X_train[np.random.choice(len(X_train), n_samples, replace=False)]
            else:
                X_train_sample = X_train
            
            if len(X_test) > n_samples:
                X_test_sample = X_test[np.random.choice(len(X_test), n_samples, replace=False)]
            else:
                X_test_sample = X_test
            
            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For models with probability outputs
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
                shap_values = self.shap_explainer.shap_values(X_test_sample)
            else:
                # For models without probability outputs
                self.shap_explainer = shap.KernelExplainer(model.predict, X_train_sample)
                shap_values = self.shap_explainer.shap_values(X_test_sample)
            
            # Calculate summary statistics
            shap_summary = {}
            
            if isinstance(shap_values, list):
                # Multi-class classification
                for i in range(len(shap_values)):
                    shap_summary[f'class_{i}'] = {
                        'mean_abs_shap': np.mean(np.abs(shap_values[i]), axis=0).tolist(),
                        'mean_shap': np.mean(shap_values[i], axis=0).tolist()
                    }
            else:
                # Single output
                shap_summary['global'] = {
                    'mean_abs_shap': np.mean(np.abs(shap_values), axis=0).tolist(),
                    'mean_shap': np.mean(shap_values, axis=0).tolist()
                }
            
            # Get top features by mean absolute SHAP
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:10]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': self.feature_names[idx],
                    'mean_abs_shap': float(mean_abs_shap[idx]),
                    'feature_index': int(idx)
                })
            
            shap_summary['top_features'] = top_features
            
            return shap_summary
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return {'error': str(e)}
    
    def _calculate_partial_dependence(self, model, X_train, features=None, grid_resolution=20):
        """Calculate partial dependence plots"""
        try:
            if features is None:
                # Use top 4 features by variance
                variances = np.var(X_train, axis=0)
                top_indices = np.argsort(variances)[::-1][:4]
                features = top_indices.tolist()
            
            pd_data = {}
            
            for feature_idx in features:
                if feature_idx >= X_train.shape[1]:
                    continue
                
                feature_name = self.feature_names[feature_idx]
                
                # Create grid of values for this feature
                feature_values = X_train[:, feature_idx]
                grid = np.linspace(np.percentile(feature_values, 5), 
                                  np.percentile(feature_values, 95), 
                                  grid_resolution)
                
                # Calculate partial dependence
                pd_values = []
                
                for value in grid:
                    X_temp = X_train.copy()
                    X_temp[:, feature_idx] = value
                    
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(X_temp)
                        # Use probability of positive class
                        if len(predictions.shape) > 1:
                            pd_values.append(np.mean(predictions[:, 1]))
                        else:
                            pd_values.append(np.mean(predictions))
                    else:
                        predictions = model.predict(X_temp)
                        pd_values.append(np.mean(predictions))
                
                pd_data[feature_name] = {
                    'feature_values': grid.tolist(),
                    'pd_values': pd_values,
                    'feature_index': int(feature_idx)
                }
            
            return pd_data
            
        except Exception as e:
            print(f"Error calculating partial dependence: {e}")
            return {'error': str(e)}
    
    def _analyze_decision_boundaries(self, model, X_train):
        """Analyze model decision boundaries"""
        try:
            # Use PCA to reduce to 2D for visualization
            from sklearn.decomposition import PCA
            
            if X_train.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_train)
                explained_variance = pca.explained_variance_ratio_.tolist()
            else:
                X_2d = X_train
                explained_variance = [1.0, 0.0] if X_train.shape[1] == 1 else [0.5, 0.5]
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_train)
                if len(predictions.shape) > 1:
                    predictions = predictions[:, 1]  # Use positive class probability
            else:
                predictions = model.predict(X_train)
            
            decision_data = {
                'X_2d': X_2d.tolist(),
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'explained_variance': explained_variance,
                'feature_1': 'PCA Component 1',
                'feature_2': 'PCA Component 2'
            }
            
            # Calculate decision boundary (simplified)
            if X_2d.shape[1] == 2:
                # Create grid for decision boundary
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                    np.linspace(y_min, y_max, 50))
                
                # Need to inverse transform from PCA space
                if X_train.shape[1] > 2:
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    grid_points = pca.inverse_transform(grid_points)
                    
                    if hasattr(model, 'predict_proba'):
                        Z = model.predict_proba(grid_points)
                        if len(Z.shape) > 1:
                            Z = Z[:, 1]
                    else:
                        Z = model.predict(grid_points)
                    
                    Z = Z.reshape(xx.shape)
                    
                    decision_data['decision_boundary'] = {
                        'xx': xx.tolist(),
                        'yy': yy.tolist(),
                        'Z': Z.tolist()
                    }
            
            return decision_data
            
        except Exception as e:
            print(f"Error analyzing decision boundaries: {e}")
            return {'error': str(e)}
    
    def _analyze_model_confidence(self, model, X_test):
        """Analyze model confidence and uncertainty"""
        try:
            confidence_data = {}
            
            if hasattr(model, 'predict_proba'):
                # Get probability predictions
                probabilities = model.predict_proba(X_test)
                
                # Calculate confidence metrics
                max_probs = np.max(probabilities, axis=1)
                
                confidence_data = {
                    'mean_confidence': float(np.mean(max_probs)),
                    'std_confidence': float(np.std(max_probs)),
                    'min_confidence': float(np.min(max_probs)),
                    'max_confidence': float(np.max(max_probs)),
                    'confidence_distribution': np.histogram(max_probs, bins=10)[0].tolist(),
                    'confidence_bins': np.histogram(max_probs, bins=10)[1].tolist(),
                    'low_confidence_samples': np.sum(max_probs < 0.7),
                    'high_confidence_samples': np.sum(max_probs > 0.9)
                }
                
                # Calculate uncertainty (entropy)
                from scipy.stats import entropy
                
                uncertainties = []
                for probs in probabilities:
                    uncertainties.append(entropy(probs))
                
                confidence_data['uncertainty'] = {
                    'mean': float(np.mean(uncertainties)),
                    'std': float(np.std(uncertainties)),
                    'distribution': np.histogram(uncertainties, bins=10)[0].tolist()
                }
            
            else:
                # For models without probability outputs
                predictions = model.predict(X_test)
                confidence_data = {
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    'note': 'Model does not provide probability outputs'
                }
            
            return confidence_data
            
        except Exception as e:
            print(f"Error analyzing model confidence: {e}")
            return {'error': str(e)}
    
    def explain_prediction(self, model, X_instance, feature_names=None):
        """Explain individual prediction"""
        try:
            if feature_names is None:
                feature_names = self.feature_names or [f'feature_{i}' for i in range(X_instance.shape[1])]
            
            explanation = {}
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_instance.reshape(1, -1))[0]
                prediction = np.argmax(probabilities)
                confidence = np.max(probabilities)
                
                explanation['prediction'] = {
                    'class': int(prediction),
                    'confidence': float(confidence),
                    'probabilities': probabilities.tolist()
                }
            else:
                prediction = model.predict(X_instance.reshape(1, -1))[0]
                explanation['prediction'] = {
                    'class': int(prediction) if hasattr(prediction, '__int__') else prediction,
                    'confidence': None,
                    'note': 'Model does not provide confidence scores'
                }
            
            # Calculate feature contributions using LIME-like approach
            feature_contributions = self._calculate_feature_contributions(model, X_instance)
            
            explanation['feature_contributions'] = []
            for i, (name, contribution) in enumerate(zip(feature_names, feature_contributions)):
                explanation['feature_contributions'].append({
                    'feature': name,
                    'contribution': float(contribution),
                    'value': float(X_instance[i]) if i < len(X_instance) else 0.0,
                    'abs_contribution': float(abs(contribution))
                })
            
            # Sort by absolute contribution
            explanation['feature_contributions'].sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            # Get top contributing features
            explanation['top_contributors'] = explanation['feature_contributions'][:10]
            
            # Generate natural language explanation
            explanation['natural_language'] = self._generate_natural_language_explanation(
                explanation['prediction'],
                explanation['top_contributors']
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def _calculate_feature_contributions(self, model, X_instance, n_samples=1000):
        """Calculate feature contributions for individual prediction"""
        # Simplified version - in practice, use LIME or SHAP
        
        # Create perturbed samples
        n_features = len(X_instance)
        contributions = np.zeros(n_features)
        
        for _ in range(n_samples):
            # Randomly select features to perturb
            mask = np.random.binomial(1, 0.5, n_features)
            perturbed = X_instance.copy()
            
            # Add small random noise to selected features
            noise = np.random.normal(0, 0.1, n_features)
            perturbed = perturbed + mask * noise
            
            # Get prediction difference
            if hasattr(model, 'predict_proba'):
                original_prob = model.predict_proba(X_instance.reshape(1, -1))[0]
                perturbed_prob = model.predict_proba(perturbed.reshape(1, -1))[0]
                diff = np.linalg.norm(original_prob - perturbed_prob)
            else:
                original_pred = model.predict(X_instance.reshape(1, -1))[0]
                perturbed_pred = model.predict(perturbed.reshape(1, -1))[0]
                diff = 1 if original_pred != perturbed_pred else 0
            
            # Attribute diff to perturbed features
            contributions += mask * diff
        
        # Normalize contributions
        if np.sum(contributions) > 0:
            contributions = contributions / np.sum(contributions)
        
        return contributions
    
    def _generate_natural_language_explanation(self, prediction, top_contributors):
        """Generate natural language explanation"""
        explanation_parts = []
        
        # Start with prediction
        if prediction.get('confidence') is not None:
            conf = prediction['confidence']
            if conf > 0.9:
                confidence_desc = "with high confidence"
            elif conf > 0.7:
                confidence_desc = "with moderate confidence"
            else:
                confidence_desc = "with low confidence"
            
            explanation_parts.append(f"The model predicts class {prediction['class']} {confidence_desc}.")
        else:
            explanation_parts.append(f"The model predicts class {prediction['class']}.")
        
        # Add feature contributions
        if top_contributors:
            explanation_parts.append("The prediction is primarily influenced by:")
            
            for i, contributor in enumerate(top_contributors[:3]):
                feature = contributor['feature']
                contribution = contributor['contribution']
                value = contributor['value']
                
                # Interpret feature
                if 'tremor' in feature.lower():
                    desc = f"tremor level ({value:.2f})"
                elif 'hesitation' in feature.lower():
                    desc = f"hesitation frequency ({value:.2f})"
                elif 'pressure' in feature.lower():
                    desc = f"writing pressure ({value:.2f})"
                elif 'velocity' in feature.lower():
                    desc = f"writing speed ({value:.2f})"
                elif 'irregularity' in feature.lower():
                    desc = f"stroke irregularity ({value:.2f})"
                else:
                    desc = f"{feature} ({value:.2f})"
                
                direction = "increased" if contribution > 0 else "decreased"
                explanation_parts.append(f"- {direction} {desc}")
        
        # Add overall interpretation
        if prediction.get('class') == 1:  # Assuming 1 is stress class
            explanation_parts.append("These patterns are commonly associated with elevated stress levels.")
        elif prediction.get('class') == 0:
            explanation_parts.append("These patterns indicate normal, relaxed handwriting.")
        
        return " ".join(explanation_parts)
    
    def create_explanation_report(self, model, X_train, X_test, y_test=None):
        """Create comprehensive explanation report"""
        report = {}
        
        # 1. Model Performance
        if y_test is not None:
            from sklearn.metrics import accuracy_score, classification_report
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            report['performance'] = {
                'accuracy': float(accuracy),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        
        # 2. Feature Analysis
        report['feature_analysis'] = self.explain_model(model, X_train, X_test)
        
        # 3. Error Analysis
        if y_test is not None:
            report['error_analysis'] = self._analyze_errors(model, X_test, y_test)
        
        # 4. Fairness Analysis
        report['fairness'] = self._analyze_fairness(model, X_test)
        
        # 5. Recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _analyze_errors(self, model, X_test, y_test):
        """Analyze model errors"""
        y_pred = model.predict(X_test)
        errors = y_pred != y_test
        
        error_analysis = {
            'error_rate': float(np.mean(errors)),
            'error_count': int(np.sum(errors)),
            'confusion': {}
        }
        
        # Analyze which classes are confused
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    error_analysis['confusion'][f'{i}_to_{j}'] = {
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / np.sum(cm[i, :]))
                    }
        
        return error_analysis
    
    def _analyze_fairness(self, model, X_test):
        """Analyze model fairness across feature ranges"""
        fairness = {
            'consistency': {},
            'bias_detection': {}
        }
        
        # Check prediction consistency across feature quantiles
        for i, feature_name in enumerate(self.feature_names[:5]):  # Check top 5 features
            feature_values = X_test[:, i]
            quantiles = np.percentile(feature_values, [25, 50, 75])
            
            predictions_by_quantile = []
            for q in range(4):
                if q == 0:
                    mask = feature_values <= quantiles[0]
                elif q == 3:
                    mask = feature_values > quantiles[2]
                else:
                    mask = (feature_values > quantiles[q-1]) & (feature_values <= quantiles[q])
                
                if np.any(mask):
                    quantile_preds = model.predict(X_test[mask])
                    if hasattr(model, 'predict_proba'):
                        quantile_probs = model.predict_proba(X_test[mask])
                        avg_prob = np.mean(quantile_probs, axis=0)
                        predictions_by_quantile.append({
                            'quantile': q,
                            'mean_prediction': float(np.mean(quantile_preds)),
                            'probability_distribution': avg_prob.tolist()
                        })
            
            fairness['consistency'][feature_name] = predictions_by_quantile
        
        return fairness
    
    def _generate_recommendations(self, report):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check feature importance stability
        if 'feature_analysis' in report and 'feature_importance' in report['feature_analysis']:
            fi = report['feature_analysis']['feature_importance']
            if 'sorted_features' in fi:
                top_features = fi['sorted_features'][:5]
                
                # Check if top features have stable importance
                importance_stds = [f['importance_std'] for f in top_features]
                if np.mean(importance_stds) > 0.1:
                    recommendations.append(
                        "Consider collecting more data to stabilize feature importance estimates."
                    )
        
        # Check model confidence
        if 'feature_analysis' in report and 'confidence_analysis' in report['feature_analysis']:
            conf = report['feature_analysis']['confidence_analysis']
            if 'mean_confidence' in conf and conf['mean_confidence'] < 0.7:
                recommendations.append(
                    "Model shows low confidence in predictions. Consider adding more discriminative features."
                )
        
        # Check error patterns
        if 'error_analysis' in report:
            error_rate = report['error_analysis']['error_rate']
            if error_rate > 0.3:
                recommendations.append(
                    f"High error rate ({error_rate:.1%}) detected. Model may need retraining with more diverse data."
                )
        
        # General recommendations
        recommendations.extend([
            "Monitor feature distributions over time to detect data drift.",
            "Regularly validate model performance on new data.",
            "Consider ensemble methods to improve robustness.",
            "Document model decisions for auditability and compliance."
        ])
        
        return recommendations
    
    def visualize_explanations(self, explanations, save_path=None):
        """Create visualization of explanations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature Importance
        if 'feature_importance' in explanations:
            fi = explanations['feature_importance']
            if 'visualization' in fi:
                vis = fi['visualization']
                axes[0, 0].barh(vis['x'][:10], vis['y'][:10], xerr=vis['error'][:10])
                axes[0, 0].set_xlabel('Importance')
                axes[0, 0].set_title('Top 10 Feature Importance')
        
        # 2. SHAP Summary
        if 'shap_values' in explanations:
            # Simplified visualization
            axes[0, 1].text(0.1, 0.5, "SHAP analysis available\nCheck detailed report", 
                           fontsize=12, ha='left', va='center')
            axes[0, 1].set_title('SHAP Analysis')
            axes[0, 1].axis('off')
        
        # 3. Decision Boundaries
        if 'decision_boundaries' in explanations:
            db = explanations['decision_boundaries']
            if 'decision_boundary' in db:
                # Create contour plot
                xx = np.array(db['decision_boundary']['xx'])
                yy = np.array(db['decision_boundary']['yy'])
                Z = np.array(db['decision_boundary']['Z'])
                
                axes[1, 0].contourf(xx, yy, Z, alpha=0.8)
                axes[1, 0].set_xlabel(db['feature_1'])
                axes[1, 0].set_ylabel(db['feature_2'])
                axes[1, 0].set_title('Decision Boundary Visualization')
        
        # 4. Confidence Distribution
        if 'confidence_analysis' in explanations:
            conf = explanations['confidence_analysis']
            if 'confidence_distribution' in conf:
                bins = conf['confidence_bins']
                dist = conf['confidence_distribution']
                
                axes[1, 1].bar(bins[:-1], dist, width=np.diff(bins))
                axes[1, 1].set_xlabel('Confidence')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Model Confidence Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig