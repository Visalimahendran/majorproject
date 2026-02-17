import numpy as np
from scipy import stats, signal, fft
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

class PatternAnalyzer:
    """Analyze patterns in handwriting for mental health assessment"""
    
    def __init__(self):
        # Pattern detection thresholds
        self.thresholds = {
            'tremor': 0.3,
            'hesitation': 0.15,
            'irregularity': 0.4,
            'pressure_variability': 0.25,
            'velocity_spikes': 0.1
        }
        
        # Mental health pattern signatures
        self.pattern_signatures = {
            'anxiety': ['high_tremor', 'high_hesitation', 'irregular_pressure'],
            'depression': ['slow_writing', 'reduced_pressure', 'simplified_strokes'],
            'stress': ['high_velocity_variability', 'pressure_spikes', 'tense_writing'],
            'fatigue': ['inconsistent_speed', 'incomplete_strokes', 'reduced_accuracy']
        }
    
    def analyze(self, features, stroke_data=None):
        """Comprehensive pattern analysis"""
        patterns = {}
        
        # 1. Detect specific patterns
        patterns.update(self._detect_tremor_patterns(features))
        patterns.update(self._detect_hesitation_patterns(features))
        patterns.update(self._detect_irregularity_patterns(features))
        patterns.update(self._detect_pressure_patterns(features))
        patterns.update(self._detect_velocity_patterns(features))
        patterns.update(self._detect_rhythm_patterns(features))
        
        # 2. Analyze stroke characteristics if stroke data provided
        if stroke_data:
            patterns.update(self._analyze_stroke_characteristics(stroke_data))
        
        # 3. Identify mental health patterns
        mental_health_patterns = self._identify_mental_health_patterns(patterns)
        patterns['mental_health_patterns'] = mental_health_patterns
        
        # 4. Calculate pattern severity scores
        severity_scores = self._calculate_severity_scores(patterns)
        patterns['severity_scores'] = severity_scores
        
        # 5. Generate pattern insights
        insights = self._generate_pattern_insights(patterns)
        patterns['insights'] = insights
        
        return patterns
    
    def _detect_tremor_patterns(self, features):
        """Detect tremor-related patterns"""
        patterns = {}
        
        # Check for tremor indicators
        tremor_indicators = [
            ('tremor_intensity', 'high_tremor'),
            ('tremor_ratio', 'significant_tremor'),
            ('spectral_tremor_power', 'tremor_power')
        ]
        
        for feature_name, pattern_name in tremor_indicators:
            if feature_name in features:
                value = features[feature_name]
                if value > self.thresholds['tremor']:
                    patterns[pattern_name] = {
                        'value': float(value),
                        'severity': self._calculate_severity(value, self.thresholds['tremor'])
                    }
        
        # Analyze tremor frequency distribution
        if 'tremor_frequency' in features:
            freq = features['tremor_frequency']
            if 3 <= freq <= 8:
                patterns['physiological_tremor'] = {'frequency': float(freq)}
            elif 8 < freq <= 12:
                patterns['enhanced_physiological_tremor'] = {'frequency': float(freq)}
            elif freq > 12:
                patterns['pathological_tremor'] = {'frequency': float(freq)}
        
        return patterns
    
    def _detect_hesitation_patterns(self, features):
        """Detect hesitation patterns"""
        patterns = {}
        
        hesitation_indicators = [
            ('hesitation_ratio', 'frequent_hesitations'),
            ('pause_count', 'multiple_pauses'),
            ('hesitation_time', 'prolonged_hesitations')
        ]
        
        for feature_name, pattern_name in hesitation_indicators:
            if feature_name in features:
                value = features[feature_name]
                if feature_name == 'hesitation_ratio' and value > self.thresholds['hesitation']:
                    patterns[pattern_name] = {
                        'value': float(value),
                        'severity': self._calculate_severity(value, self.thresholds['hesitation'])
                    }
                elif feature_name == 'pause_count' and value > 5:
                    patterns[pattern_name] = {'count': int(value)}
                elif feature_name == 'hesitation_time' and value > 2.0:
                    patterns[pattern_name] = {'total_time': float(value)}
        
        # Micro-hesitations
        if 'micro_pause_count' in features and features['micro_pause_count'] > 3:
            patterns['micro_hesitations'] = {'count': int(features['micro_pause_count'])}
        
        return patterns
    
    def _detect_irregularity_patterns(self, features):
        """Detect irregularity patterns"""
        patterns = {}
        
        irregularity_indicators = [
            ('irregularity_index', 'high_irregularity'),
            ('jerk_cost', 'jerky_movements'),
            ('velocity_std', 'inconsistent_speed'),
            ('fractal_dimension', 'complex_patterns')
        ]
        
        for feature_name, pattern_name in irregularity_indicators:
            if feature_name in features:
                value = features[feature_name]
                
                if feature_name == 'irregularity_index' and value > self.thresholds['irregularity']:
                    patterns[pattern_name] = {
                        'value': float(value),
                        'severity': self._calculate_severity(value, self.thresholds['irregularity'])
                    }
                elif feature_name == 'jerk_cost' and value > 0.5:
                    patterns[pattern_name] = {'jerk_cost': float(value)}
                elif feature_name == 'velocity_std' and value > 0.2:
                    patterns[pattern_name] = {'speed_variability': float(value)}
                elif feature_name == 'fractal_dimension' and value > 1.5:
                    patterns['complex_strokes'] = {'fractal_dim': float(value)}
                elif feature_name == 'fractal_dimension' and value < 1.2:
                    patterns['simplified_strokes'] = {'fractal_dim': float(value)}
        
        return patterns
    
    def _detect_pressure_patterns(self, features):
        """Detect pressure-related patterns"""
        patterns = {}
        
        pressure_indicators = [
            ('pressure_std', 'irregular_pressure'),
            ('pressure_cv', 'pressure_variability'),
            ('pressure_spikes', 'pressure_spikes'),
            ('pressure_gradient_mean', 'rapid_pressure_changes')
        ]
        
        for feature_name, pattern_name in pressure_indicators:
            if feature_name in features:
                value = features[feature_name]
                
                if feature_name in ['pressure_std', 'pressure_cv'] and value > self.thresholds['pressure_variability']:
                    patterns[pattern_name] = {
                        'value': float(value),
                        'severity': self._calculate_severity(value, self.thresholds['pressure_variability'])
                    }
                elif feature_name == 'pressure_spikes' and value > 0:
                    patterns[pattern_name] = {'spike_count': int(value)}
                elif feature_name == 'pressure_gradient_mean' and value > 0.1:
                    patterns[pattern_name] = {'gradient': float(value)}
        
        # Overall pressure level
        if 'pressure_mean' in features:
            pressure_mean = features['pressure_mean']
            if pressure_mean < 0.3:
                patterns['light_pressure'] = {'mean_pressure': float(pressure_mean)}
            elif pressure_mean > 0.7:
                patterns['heavy_pressure'] = {'mean_pressure': float(pressure_mean)}
            else:
                patterns['normal_pressure'] = {'mean_pressure': float(pressure_mean)}
        
        return patterns
    
    def _detect_velocity_patterns(self, features):
        """Detect velocity-related patterns"""
        patterns = {}
        
        velocity_indicators = [
            ('velocity_spike_ratio', 'velocity_spikes'),
            ('velocity_skewness', 'asymmetric_velocity'),
            ('acceleration_mean', 'rapid_accelerations'),
            ('jerk_mean', 'jerky_movements')
        ]
        
        for feature_name, pattern_name in velocity_indicators:
            if feature_name in features:
                value = features[feature_name]
                
                if feature_name == 'velocity_spike_ratio' and value > self.thresholds['velocity_spikes']:
                    patterns[pattern_name] = {
                        'value': float(value),
                        'severity': self._calculate_severity(value, self.thresholds['velocity_spikes'])
                    }
                elif feature_name == 'velocity_skewness' and abs(value) > 1.0:
                    patterns[pattern_name] = {'skewness': float(value)}
                elif feature_name == 'acceleration_mean' and value > 0.5:
                    patterns[pattern_name] = {'mean_acceleration': float(value)}
                elif feature_name == 'jerk_mean' and value > 0.3:
                    patterns[pattern_name] = {'mean_jerk': float(value)}
        
        # Overall speed pattern
        if 'velocity_mean' in features:
            speed = features['velocity_mean']
            if speed < 0.1:
                patterns['slow_writing'] = {'mean_speed': float(speed)}
            elif speed > 0.5:
                patterns['fast_writing'] = {'mean_speed': float(speed)}
        
        return patterns
    
    def _detect_rhythm_patterns(self, features):
        """Detect rhythm and timing patterns"""
        patterns = {}
        
        rhythm_indicators = [
            ('rhythm_regularity', 'irregular_rhythm'),
            ('velocity_autocorr_time', 'short_correlation'),
            ('spectral_entropy', 'complex_spectrum')
        ]
        
        for feature_name, pattern_name in rhythm_indicators:
            if feature_name in features:
                value = features[feature_name]
                
                if feature_name == 'rhythm_regularity' and value < 0.7:
                    patterns[pattern_name] = {'regularity': float(value)}
                elif feature_name == 'velocity_autocorr_time' and value < 5:
                    patterns[pattern_name] = {'correlation_time': float(value)}
                elif feature_name == 'spectral_entropy' and value > 2.0:
                    patterns[pattern_name] = {'spectral_entropy': float(value)}
        
        # Pause distribution pattern
        if 'pause_expon_scale' in features:
            scale = features['pause_expon_scale']
            if scale > 0.5:
                patterns['long_pauses'] = {'scale_parameter': float(scale)}
            elif scale < 0.1:
                patterns['short_frequent_pauses'] = {'scale_parameter': float(scale)}
        
        return patterns
    
    def _analyze_stroke_characteristics(self, stroke_data):
        """Analyze stroke-level characteristics"""
        patterns = {}
        
        if not stroke_data or len(stroke_data) == 0:
            return patterns
        
        # Analyze stroke statistics
        stroke_lengths = []
        stroke_durations = []
        stroke_speeds = []
        
        for stroke in stroke_data:
            if len(stroke) >= 2:
                # Calculate stroke length
                if isinstance(stroke[0], dict):
                    x = [p.get('x', 0) for p in stroke]
                    y = [p.get('y', 0) for p in stroke]
                else:
                    x = [p[0] for p in stroke]
                    y = [p[1] for p in stroke]
                
                length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
                stroke_lengths.append(length)
                
                # Calculate stroke duration
                if 'timestamp' in stroke[0] if isinstance(stroke[0], dict) else False:
                    timestamps = [p['timestamp'] for p in stroke]
                    if hasattr(timestamps[0], 'timestamp'):
                        timestamps = [ts.timestamp() for ts in timestamps]
                    duration = timestamps[-1] - timestamps[0]
                else:
                    duration = len(stroke) / 100  # Approximate
                
                stroke_durations.append(duration)
                
                # Calculate average speed
                if duration > 0:
                    stroke_speeds.append(length / duration)
        
        if stroke_lengths:
            # Stroke length patterns
            mean_length = np.mean(stroke_lengths)
            std_length = np.std(stroke_lengths)
            
            if std_length / mean_length > 0.5:
                patterns['variable_stroke_lengths'] = {
                    'cv': float(std_length / mean_length)
                }
            
            if mean_length < 0.1:
                patterns['short_strokes'] = {'mean_length': float(mean_length)}
            elif mean_length > 0.5:
                patterns['long_strokes'] = {'mean_length': float(mean_length)}
        
        if stroke_durations:
            # Stroke duration patterns
            mean_duration = np.mean(stroke_durations)
            if mean_duration < 0.1:
                patterns['brief_strokes'] = {'mean_duration': float(mean_duration)}
            elif mean_duration > 0.5:
                patterns['prolonged_strokes'] = {'mean_duration': float(mean_duration)}
        
        if stroke_speeds:
            # Stroke speed consistency
            speed_cv = np.std(stroke_speeds) / np.mean(stroke_speeds)
            if speed_cv > 0.5:
                patterns['inconsistent_stroke_speeds'] = {'speed_cv': float(speed_cv)}
        
        return patterns
    
    def _identify_mental_health_patterns(self, patterns):
        """Identify mental health-related patterns"""
        mental_health_patterns = defaultdict(list)
        
        # Map detected patterns to mental health conditions
        for pattern_name, pattern_data in patterns.items():
            if pattern_name in ['high_tremor', 'significant_tremor', 'physiological_tremor']:
                mental_health_patterns['anxiety'].append(pattern_name)
                mental_health_patterns['stress'].append(pattern_name)
            
            if pattern_name in ['frequent_hesitations', 'multiple_pauses', 'micro_hesitations']:
                mental_health_patterns['anxiety'].append(pattern_name)
                mental_health_patterns['indecision'].append(pattern_name)
            
            if pattern_name in ['high_irregularity', 'jerky_movements', 'inconsistent_speed']:
                mental_health_patterns['stress'].append(pattern_name)
                mental_health_patterns['fatigue'].append(pattern_name)
            
            if pattern_name in ['irregular_pressure', 'pressure_spikes', 'rapid_pressure_changes']:
                mental_health_patterns['stress'].append(pattern_name)
                mental_health_patterns['tension'].append(pattern_name)
            
            if pattern_name in ['slow_writing', 'simplified_strokes', 'reduced_pressure']:
                mental_health_patterns['depression'].append(pattern_name)
                mental_health_patterns['fatigue'].append(pattern_name)
            
            if pattern_name in ['fast_writing', 'velocity_spikes', 'rapid_accelerations']:
                mental_health_patterns['anxiety'].append(pattern_name)
                mental_health_patterns['agitation'].append(pattern_name)
        
        # Calculate confidence scores for each condition
        confidence_scores = {}
        for condition, indicators in mental_health_patterns.items():
            # Simple confidence based on number of matching indicators
            total_indicators = len(self.pattern_signatures.get(condition, []))
            matched_indicators = len(set(indicators))
            
            if total_indicators > 0:
                confidence = matched_indicators / total_indicators
                confidence_scores[condition] = {
                    'confidence': float(confidence),
                    'matched_indicators': matched_indicators,
                    'total_indicators': total_indicators,
                    'indicators': list(set(indicators))
                }
        
        return confidence_scores
    
    def _calculate_severity_scores(self, patterns):
        """Calculate severity scores for detected patterns"""
        severity_scores = {}
        
        # Calculate overall pattern severity
        pattern_severities = []
        
        for pattern_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'severity' in pattern_data:
                pattern_severities.append(pattern_data['severity'])
            elif pattern_name in ['high_tremor', 'frequent_hesitations', 'high_irregularity']:
                # Estimate severity from pattern presence
                pattern_severities.append(0.5)
        
        if pattern_severities:
            overall_severity = np.mean(pattern_severities)
        else:
            overall_severity = 0.0
        
        severity_scores['overall'] = float(overall_severity)
        
        # Calculate domain-specific severity
        domains = {
            'motor_control': ['high_tremor', 'jerky_movements', 'inconsistent_speed'],
            'cognitive': ['frequent_hesitations', 'multiple_pauses', 'micro_hesitations'],
            'emotional': ['pressure_spikes', 'irregular_pressure', 'tense_writing'],
            'fatigue': ['slow_writing', 'simplified_strokes', 'incomplete_strokes']
        }
        
        for domain, domain_patterns in domains.items():
            domain_severity = 0.0
            count = 0
            
            for pattern in domain_patterns:
                if pattern in patterns:
                    if 'severity' in patterns[pattern]:
                        domain_severity += patterns[pattern]['severity']
                        count += 1
                    else:
                        domain_severity += 0.3
                        count += 1
            
            if count > 0:
                severity_scores[domain] = float(domain_severity / count)
            else:
                severity_scores[domain] = 0.0
        
        return severity_scores
    
    def _generate_pattern_insights(self, patterns):
        """Generate human-readable insights from patterns"""
        insights = []
        
        # Overall assessment
        severity = patterns.get('severity_scores', {}).get('overall', 0)
        
        if severity < 0.3:
            insights.append("Writing patterns appear normal and consistent.")
        elif severity < 0.6:
            insights.append("Mild deviations detected in writing patterns.")
        elif severity < 0.8:
            insights.append("Moderate abnormalities observed in handwriting.")
        else:
            insights.append("Significant irregularities detected in writing patterns.")
        
        # Specific pattern insights
        if 'high_tremor' in patterns:
            insights.append("Noticeable tremor detected, which may indicate nervous system arousal.")
        
        if 'frequent_hesitations' in patterns:
            insights.append("Frequent hesitations suggest possible cognitive load or uncertainty.")
        
        if 'irregular_pressure' in patterns:
            insights.append("Inconsistent pressure application may reflect emotional variability.")
        
        if 'jerky_movements' in patterns:
            insights.append("Jerky stroke movements could indicate motor control issues.")
        
        if 'slow_writing' in patterns:
            insights.append("Slow writing speed may suggest reduced energy or motivation.")
        
        if 'fast_writing' in patterns:
            insights.append("Rapid writing may indicate agitation or time pressure.")
        
        # Mental health pattern insights
        mental_health = patterns.get('mental_health_patterns', {})
        
        for condition, data in mental_health.items():
            confidence = data.get('confidence', 0)
            if confidence > 0.6:
                if condition == 'anxiety':
                    insights.append("Patterns consistent with anxiety-related handwriting characteristics.")
                elif condition == 'stress':
                    insights.append("Writing patterns show signs of stress-related tension.")
                elif condition == 'depression':
                    insights.append("Characteristics suggestive of depressive symptomatology.")
                elif condition == 'fatigue':
                    insights.append("Patterns indicate possible fatigue or reduced mental energy.")
        
        return insights
    
    def _calculate_severity(self, value, threshold):
        """Calculate severity score based on deviation from threshold"""
        if value <= threshold:
            return 0.0
        
        # Normalized severity (0-1)
        severity = min(1.0, (value - threshold) / (threshold * 2))
        return severity
    
    def track_pattern_changes(self, pattern_history):
        """Track changes in patterns over time"""
        if len(pattern_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Extract severity scores over time
        severities = [ph.get('severity_scores', {}).get('overall', 0) 
                     for ph in pattern_history]
        
        # Calculate trend
        x = np.arange(len(severities))
        slope, intercept = np.polyfit(x, severities, 1)
        
        if slope > 0.05:
            trend = 'worsening'
        elif slope < -0.05:
            trend = 'improving'
        else:
            trend = 'stable'
        
        # Detect significant changes
        changes = []
        for i in range(1, len(pattern_history)):
            prev = pattern_history[i-1]
            curr = pattern_history[i]
            
            # Compare pattern presence
            prev_patterns = set(prev.keys())
            curr_patterns = set(curr.keys())
            
            new_patterns = curr_patterns - prev_patterns
            resolved_patterns = prev_patterns - curr_patterns
            
            if new_patterns:
                changes.append({
                    'time_point': i,
                    'new_patterns': list(new_patterns),
                    'type': 'emerging'
                })
            
            if resolved_patterns:
                changes.append({
                    'time_point': i,
                    'resolved_patterns': list(resolved_patterns),
                    'type': 'resolving'
                })
        
        return {
            'trend': trend,
            'slope': float(slope),
            'mean_severity': float(np.mean(severities)),
            'severity_std': float(np.std(severities)),
            'changes': changes,
            'severity_timeline': severities
        }