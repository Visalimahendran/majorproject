import numpy as np
import cv2
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

class RealTimeAnalysisPipeline:
    """Real-time analysis pipeline for handwriting data"""
    
    def __init__(self):
        # Feature buffers
        self.feature_buffer = []
        self.max_buffer_size = 100
        
        # Analysis parameters
        self.tremor_freq_range = (3, 12)  # Hz
        self.sampling_rate = 30  # Hz for video
        
    def analyze_writing_frames(self, frames):
        """Analyze writing frames for neuro-motor features"""
        if not frames:
            return None
        
        features = {}
        
        # Extract features from each frame
        all_tremors = []
        all_pressures = []
        all_velocities = []
        
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Extract frame features
            frame_features = self._extract_frame_features(gray, i)
            
            # Aggregate features
            if 'tremor' in frame_features:
                all_tremors.append(frame_features['tremor'])
            if 'pressure' in frame_features:
                all_pressures.append(frame_features['pressure'])
            if 'velocity' in frame_features:
                all_velocities.append(frame_features['velocity'])
        
        # Calculate aggregated features
        if all_tremors:
            features['tremor_mean'] = np.mean(all_tremors)
            features['tremor_std'] = np.std(all_tremors)
            features['tremor_intensity'] = features['tremor_mean'] * 100
        
        if all_pressures:
            features['pressure_mean'] = np.mean(all_pressures)
            features['pressure_std'] = np.std(all_pressures)
            features['pressure_variability'] = features['pressure_std'] / (features['pressure_mean'] + 1e-10)
        
        if all_velocities:
            features['velocity_mean'] = np.mean(all_velocities)
            features['velocity_std'] = np.std(all_velocities)
            features['velocity_cv'] = features['velocity_std'] / (features['velocity_mean'] + 1e-10)
        
        # Calculate Neural Pressure Index (NPI)
        if features:
            features['npi_score'] = self._calculate_npi(features)
            features['npi_category'] = self._categorize_npi(features['npi_score'])
        
        # Add to buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.max_buffer_size:
            self.feature_buffer = self.feature_buffer[-self.max_buffer_size:]
        
        return features
    
    def _extract_frame_features(self, frame, frame_index):
        """Extract features from a single frame"""
        features = {}
        
        # Basic image features
        features['brightness'] = np.mean(frame) / 255.0
        features['contrast'] = np.std(frame) / 255.0
        
        # Edge features (related to writing clarity)
        edges = cv2.Canny(frame, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            features['contour_count'] = len(contours)
            features['avg_contour_area'] = np.mean(areas) if areas else 0
            features['contour_area_std'] = np.std(areas) if len(areas) > 1 else 0
        
        # Simulate tremor (for demo purposes)
        features['tremor'] = np.clip(np.random.normal(0.15, 0.05), 0, 1)
        
        # Simulate pressure (for demo purposes)
        features['pressure'] = np.clip(np.random.normal(0.5, 0.2), 0, 1)
        
        # Simulate velocity (for demo purposes)
        features['velocity'] = np.clip(np.random.normal(0.3, 0.1), 0, 1)
        
        return features
    
    def _calculate_npi(self, features):
        """Calculate Neural Pressure Index from features"""
        npi = 0
        
        # Weight contributions from different features
        weights = {
            'tremor_mean': 0.3,
            'tremor_intensity': 0.2,
            'pressure_variability': 0.25,
            'velocity_cv': 0.15,
            'edge_density': 0.1
        }
        
        for feature, weight in weights.items():
            if feature in features:
                value = features[feature]
                # Normalize to 0-100 range
                if feature == 'tremor_mean':
                    normalized = value * 200  # Scale up
                elif feature == 'tremor_intensity':
                    normalized = value  # Already in 0-100
                else:
                    normalized = value * 100  # Scale 0-1 to 0-100
                
                npi += normalized * weight
        
        # Clip to 0-100 range
        npi = np.clip(npi, 0, 100)
        return npi
    
    def _categorize_npi(self, npi_score):
        """Categorize NPI score"""
        if npi_score < 30:
            return "Low"
        elif npi_score < 60:
            return "Moderate"
        else:
            return "High"
    
    def analyze_stroke_data(self, stroke_data):
        """Analyze digital stroke data"""
        if not stroke_data or 'strokes' not in stroke_data:
            return None
        
        strokes = stroke_data['strokes']
        features = {
            'stroke_count': len(strokes),
            'total_points': sum(len(stroke) for stroke in strokes),
            'writing_duration': stroke_data.get('duration', 0),
        }
        
        if not strokes:
            return features
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(strokes)
        features.update(temporal_features)
        
        # Extract spatial features
        spatial_features = self._extract_spatial_features(strokes)
        features.update(spatial_features)
        
        # Extract pressure features
        pressure_features = self._extract_pressure_features(strokes)
        features.update(pressure_features)
        
        # Calculate NPI
        features['npi_score'] = self._calculate_stroke_npi(features)
        features['npi_category'] = self._categorize_npi(features['npi_score'])
        
        return features
    
    def _extract_temporal_features(self, strokes):
        """Extract temporal features from strokes"""
        features = {}
        
        # Collect all timestamps
        all_timestamps = []
        for stroke in strokes:
            for point in stroke:
                if 'timestamp' in point:
                    all_timestamps.append(point['timestamp'])
        
        if len(all_timestamps) < 2:
            return features
        
        all_timestamps = np.array(all_timestamps)
        
        # Calculate time intervals
        time_intervals = np.diff(all_timestamps)
        
        # Remove zero or negative intervals
        time_intervals = time_intervals[time_intervals > 0]
        
        if len(time_intervals) == 0:
            return features
        
        # Basic statistics
        features['mean_interval'] = np.mean(time_intervals)
        features['std_interval'] = np.std(time_intervals)
        features['interval_cv'] = features['std_interval'] / features['mean_interval']
        
        # Detect pauses (long intervals)
        pause_threshold = np.percentile(time_intervals, 75)  # Top 25% as pauses
        pauses = time_intervals[time_intervals > pause_threshold]
        features['pause_count'] = len(pauses)
        features['pause_ratio'] = len(pauses) / len(time_intervals)
        
        # Calculate writing rhythm
        if len(time_intervals) > 10:
            # Autocorrelation of intervals
            autocorr = np.correlate(time_intervals - features['mean_interval'], 
                                   time_intervals - features['mean_interval'], 
                                   mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            
            # Find first zero crossing
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                features['rhythm_consistency'] = 1.0 / (1.0 + zero_crossings[0])
            else:
                features['rhythm_consistency'] = 0.5
        else:
            features['rhythm_consistency'] = 0.5
        
        return features
    
    def _extract_spatial_features(self, strokes):
        """Extract spatial features from strokes"""
        features = {}
        
        # Collect all points
        all_points = []
        for stroke in strokes:
            for point in stroke:
                if 'x' in point and 'y' in point:
                    all_points.append((point['x'], point['y']))
        
        if len(all_points) < 10:
            return features
        
        points_array = np.array(all_points)
        
        # Basic spatial statistics
        features['x_range'] = np.ptp(points_array[:, 0])
        features['y_range'] = np.ptp(points_array[:, 1])
        features['writing_size'] = np.sqrt(features['x_range'] ** 2 + features['y_range'] ** 2)
        
        # Calculate path length
        path_length = 0
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            for i in range(1, len(stroke)):
                if 'x' in stroke[i] and 'y' in stroke[i] and 'x' in stroke[i-1] and 'y' in stroke[i-1]:
                    dx = stroke[i]['x'] - stroke[i-1]['x']
                    dy = stroke[i]['y'] - stroke[i-1]['y']
                    path_length += np.sqrt(dx**2 + dy**2)
        
        features['path_length'] = path_length
        
        # Calculate straightness (how direct is the writing)
        if len(points_array) > 1:
            start_point = points_array[0]
            end_point = points_array[-1]
            straight_distance = np.sqrt(np.sum((end_point - start_point) ** 2))
            if path_length > 0:
                features['straightness'] = straight_distance / path_length
            else:
                features['straightness'] = 0
        else:
            features['straightness'] = 0
        
        # Calculate curvature
        if len(points_array) > 2:
            # Simplified curvature calculation
            curvatures = []
            for i in range(1, len(points_array) - 1):
                p1 = points_array[i-1]
                p2 = points_array[i]
                p3 = points_array[i+1]
                
                # Calculate angle change
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    curvatures.append(angle)
            
            if curvatures:
                features['mean_curvature'] = np.mean(curvatures)
                features['curvature_std'] = np.std(curvatures)
        
        return features
    
    def _extract_pressure_features(self, strokes):
        """Extract pressure-related features"""
        features = {}
        
        # Collect all pressure values
        all_pressures = []
        for stroke in strokes:
            for point in stroke:
                if 'pressure' in point:
                    all_pressures.append(point['pressure'])
        
        if len(all_pressures) < 2:
            return features
        
        pressures = np.array(all_pressures)
        
        # Basic statistics
        features['pressure_mean'] = np.mean(pressures)
        features['pressure_std'] = np.std(pressures)
        features['pressure_cv'] = features['pressure_std'] / (features['pressure_mean'] + 1e-10)
        
        # Pressure changes
        pressure_changes = np.diff(pressures)
        features['mean_pressure_change'] = np.mean(np.abs(pressure_changes))
        features['max_pressure_change'] = np.max(np.abs(pressure_changes))
        
        # Detect pressure spikes
        spike_threshold = features['pressure_mean'] + 2 * features['pressure_std']
        spikes = pressures[pressures > spike_threshold]
        features['pressure_spike_count'] = len(spikes)
        features['pressure_spike_ratio'] = len(spikes) / len(pressures)
        
        return features
    
    def _calculate_stroke_npi(self, features):
        """Calculate NPI from stroke features"""
        npi = 0
        
        # Weight contributions
        weights = {
            'interval_cv': 0.2,          # Temporal irregularity
            'pause_ratio': 0.15,         # Hesitation
            'rhythm_consistency': -0.1,  # Negative weight - more consistency is better
            'curvature_std': 0.15,       # Stroke irregularity
            'pressure_cv': 0.2,          # Pressure variability
            'pressure_spike_ratio': 0.2   # Pressure instability
        }
        
        for feature, weight in weights.items():
            if feature in features:
                value = features[feature]
                
                # Normalize and scale
                if feature == 'rhythm_consistency':
                    # Higher consistency = lower NPI
                    normalized = (1 - value) * 100
                else:
                    normalized = min(value * 100, 100)
                
                npi += normalized * weight
        
        # Clip to 0-100 range
        npi = np.clip(npi, 0, 100)
        return npi
    
    def get_trend_analysis(self):
        """Analyze trends in feature buffer"""
        if len(self.feature_buffer) < 10:
            return None
        
        # Extract NPI scores over time
        npi_scores = [f.get('npi_score', 50) for f in self.feature_buffer if 'npi_score' in f]
        
        if len(npi_scores) < 5:
            return None
        
        # Calculate trend
        x = np.arange(len(npi_scores))
        slope, intercept = np.polyfit(x, npi_scores, 1)
        
        # Determine trend direction
        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            'trend': trend,
            'slope': float(slope),
            'mean_npi': float(np.mean(npi_scores)),
            'std_npi': float(np.std(npi_scores)),
            'current_npi': float(npi_scores[-1]),
            'samples': len(npi_scores)
        }