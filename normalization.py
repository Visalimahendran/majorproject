import numpy as np
from scipy import signal, interpolate
import warnings
warnings.filterwarnings('ignore')

class Normalizer:
    """Normalize handwriting data for consistent analysis"""
    
    def __init__(self, pressure_range=(0, 8192), speed_range=(0, 1000)):
        self.pressure_range = pressure_range
        self.speed_range = speed_range
        
    def normalize_stroke(self, stroke, target_points=100):
        """Normalize a single stroke"""
        if len(stroke) < 2:
            return stroke
        
        # Extract data
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        t = np.array([p.get('timestamp', i) for i, p in enumerate(stroke)])
        pressures = np.array([p.get('pressure', 0.5) for p in stroke])
        
        # Convert timestamps to seconds if needed
        if hasattr(t[0], 'timestamp'):
            t = np.array([ts.timestamp() for ts in t])
        
        # 1. Normalize time to start at 0
        t = t - t[0]
        
        # 2. Resample to target number of points
        x_norm, y_norm, t_norm, p_norm = self._resample_stroke(
            x, y, t, pressures, target_points
        )
        
        # 3. Normalize spatial coordinates
        x_norm, y_norm = self._normalize_spatial(x_norm, y_norm)
        
        # 4. Normalize pressure
        p_norm = self._normalize_pressure(p_norm)
        
        # 5. Calculate and normalize velocities
        vx, vy, speed = self._calculate_velocity(x_norm, y_norm, t_norm)
        speed_norm = self._normalize_speed(speed)
        
        # 6. Calculate acceleration
        ax, ay, acceleration = self._calculate_acceleration(vx, vy, t_norm)
        
        # Create normalized stroke
        normalized_stroke = []
        for i in range(len(x_norm)):
            point = {
                'x': float(x_norm[i]),
                'y': float(y_norm[i]),
                't': float(t_norm[i]),
                'pressure': float(p_norm[i]),
                'vx': float(vx[i]) if i < len(vx) else 0.0,
                'vy': float(vy[i]) if i < len(vy) else 0.0,
                'speed': float(speed_norm[i]) if i < len(speed_norm) else 0.0,
                'ax': float(ax[i]) if i < len(ax) else 0.0,
                'ay': float(ay[i]) if i < len(ay) else 0.0,
                'acceleration': float(acceleration[i]) if i < len(acceleration) else 0.0,
                'curvature': self._calculate_curvature(x_norm, y_norm, i) if i > 0 and i < len(x_norm)-1 else 0.0
            }
            normalized_stroke.append(point)
        
        return normalized_stroke
    
    def _resample_stroke(self, x, y, t, p, target_points):
        """Resample stroke to have consistent number of points"""
        # Calculate cumulative distance along stroke
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.sqrt(dx**2 + dy**2)
        cum_dist = np.concatenate(([0], np.cumsum(dist)))
        
        # Normalize cumulative distance to [0, 1]
        if cum_dist[-1] == 0:
            cum_dist_norm = np.linspace(0, 1, len(x))
        else:
            cum_dist_norm = cum_dist / cum_dist[-1]
        
        # Create interpolation functions
        try:
            fx = interpolate.interp1d(cum_dist_norm, x, kind='cubic', fill_value="extrapolate")
            fy = interpolate.interp1d(cum_dist_norm, y, kind='cubic', fill_value="extrapolate")
            ft = interpolate.interp1d(cum_dist_norm, t, kind='linear', fill_value="extrapolate")
            fp = interpolate.interp1d(cum_dist_norm, p, kind='linear', fill_value="extrapolate")
        except:
            # Fallback to linear interpolation
            fx = interpolate.interp1d(cum_dist_norm, x, kind='linear', fill_value="extrapolate")
            fy = interpolate.interp1d(cum_dist_norm, y, kind='linear', fill_value="extrapolate")
            ft = interpolate.interp1d(cum_dist_norm, t, kind='linear', fill_value="extrapolate")
            fp = interpolate.interp1d(cum_dist_norm, p, kind='linear', fill_value="extrapolate")
        
        # Generate new points
        new_dist_norm = np.linspace(0, 1, target_points)
        x_resampled = fx(new_dist_norm)
        y_resampled = fy(new_dist_norm)
        t_resampled = ft(new_dist_norm)
        p_resampled = fp(new_dist_norm)
        
        return x_resampled, y_resampled, t_resampled, p_resampled
    
    def _normalize_spatial(self, x, y):
        """Normalize spatial coordinates to unit bounding box"""
        # Translate to origin
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        if x_max - x_min == 0:
            x_norm = np.zeros_like(x)
        else:
            x_norm = (x - x_min) / (x_max - x_min)
        
        if y_max - y_min == 0:
            y_norm = np.zeros_like(y)
        else:
            y_norm = (y - y_min) / (y_max - y_min)
        
        # Center around 0.5
        x_norm = x_norm - 0.5
        y_norm = y_norm - 0.5
        
        return x_norm, y_norm
    
    def _normalize_pressure(self, pressure):
        """Normalize pressure values"""
        pressure = np.array(pressure)
        
        # Clip to range
        pressure = np.clip(pressure, self.pressure_range[0], self.pressure_range[1])
        
        # Normalize to [0, 1]
        if self.pressure_range[1] - self.pressure_range[0] > 0:
            pressure_norm = (pressure - self.pressure_range[0]) / (self.pressure_range[1] - self.pressure_range[0])
        else:
            pressure_norm = pressure
        
        return pressure_norm
    
    def _normalize_speed(self, speed):
        """Normalize speed values"""
        speed = np.array(speed)
        
        # Clip to range
        speed = np.clip(speed, self.speed_range[0], self.speed_range[1])
        
        # Normalize to [0, 1]
        if self.speed_range[1] - self.speed_range[0] > 0:
            speed_norm = speed / self.speed_range[1]
        else:
            speed_norm = speed
        
        return speed_norm
    
    def _calculate_velocity(self, x, y, t):
        """Calculate velocity from position and time"""
        dt = np.diff(t)
        
        # Avoid division by zero
        dt[dt == 0] = 1e-10
        
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Pad to match original length
        vx = np.concatenate(([vx[0]], vx))
        vy = np.concatenate(([vy[0]], vy))
        
        speed = np.sqrt(vx**2 + vy**2)
        
        return vx, vy, speed
    
    def _calculate_acceleration(self, vx, vy, t):
        """Calculate acceleration from velocity and time"""
        dt = np.diff(t)
        
        # Avoid division by zero
        dt[dt == 0] = 1e-10
        
        ax = np.diff(vx) / dt
        ay = np.diff(vy) / dt
        
        # Pad to match original length
        ax = np.concatenate(([ax[0]], ax))
        ay = np.concatenate(([ay[0]], ay))
        
        acceleration = np.sqrt(ax**2 + ay**2)
        
        return ax, ay, acceleration
    
    def _calculate_curvature(self, x, y, i):
        """Calculate curvature at point i"""
        if i == 0 or i == len(x) - 1:
            return 0.0
        
        # Get three points
        x0, y0 = x[i-1], y[i-1]
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        
        # Calculate vectors
        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])
        
        # Calculate angle between vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Curvature is change in angle
        curvature = theta
        
        return float(curvature)
    
    def normalize_dataset(self, dataset, target_points=100):
        """Normalize entire dataset"""
        normalized_dataset = []
        
        for sample in dataset:
            if 'strokes' in sample:
                normalized_strokes = []
                for stroke in sample['strokes']:
                    norm_stroke = self.normalize_stroke(stroke, target_points)
                    normalized_strokes.append(norm_stroke)
                
                normalized_sample = sample.copy()
                normalized_sample['strokes'] = normalized_strokes
                normalized_dataset.append(normalized_sample)
        
        return normalized_dataset
    
    def extract_normalized_features(self, normalized_stroke):
        """Extract features from normalized stroke"""
        if not normalized_stroke:
            return {}
        
        # Convert to arrays
        x = np.array([p['x'] for p in normalized_stroke])
        y = np.array([p['y'] for p in normalized_stroke])
        pressure = np.array([p['pressure'] for p in normalized_stroke])
        speed = np.array([p['speed'] for p in normalized_stroke])
        curvature = np.array([p['curvature'] for p in normalized_stroke])
        
        features = {}
        
        # Basic statistics
        features['stroke_length'] = len(normalized_stroke)
        features['total_distance'] = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        # Pressure features
        features['pressure_mean'] = np.mean(pressure)
        features['pressure_std'] = np.std(pressure)
        features['pressure_max'] = np.max(pressure)
        features['pressure_min'] = np.min(pressure)
        features['pressure_range'] = features['pressure_max'] - features['pressure_min']
        
        # Speed features
        features['speed_mean'] = np.mean(speed)
        features['speed_std'] = np.std(speed)
        features['speed_max'] = np.max(speed)
        features['speed_min'] = np.min(speed)
        features['speed_variation'] = features['speed_std'] / (features['speed_mean'] + 1e-10)
        
        # Curvature features
        features['curvature_mean'] = np.mean(curvature)
        features['curvature_std'] = np.std(curvature)
        features['curvature_max'] = np.max(curvature)
        
        # Spatial features
        features['x_range'] = np.max(x) - np.min(x)
        features['y_range'] = np.max(y) - np.min(y)
        features['aspect_ratio'] = features['x_range'] / (features['y_range'] + 1e-10)
        
        # Frequency domain features (for tremor analysis)
        if len(speed) > 10:
            try:
                # Remove DC component
                speed_detrended = signal.detrend(speed)
                
                # Calculate FFT
                fft_values = np.fft.fft(speed_detrended)
                frequencies = np.fft.fftfreq(len(speed_detrended))
                
                # Get power spectrum
                power_spectrum = np.abs(fft_values)**2
                
                # Find dominant frequency
                dominant_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                features['dominant_frequency'] = abs(frequencies[dominant_idx])
                features['dominant_power'] = power_spectrum[dominant_idx]
                
                # Calculate tremor index (power in 3-12 Hz range)
                tremor_mask = (abs(frequencies) >= 3/100) & (abs(frequencies) <= 12/100)
                tremor_power = np.sum(power_spectrum[tremor_mask])
                total_power = np.sum(power_spectrum[1:])
                features['tremor_index'] = tremor_power / (total_power + 1e-10)
            except:
                features['dominant_frequency'] = 0
                features['dominant_power'] = 0
                features['tremor_index'] = 0
        
        # Jerk (derivative of acceleration)
        if len(speed) > 2:
            acceleration = np.array([p['acceleration'] for p in normalized_stroke])
            jerk = np.diff(acceleration) / np.diff(np.arange(len(acceleration)))
            features['jerk_mean'] = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0
        
        # Pause detection
        speed_threshold = 0.01
        pause_mask = speed < speed_threshold
        pause_segments = self._find_segments(pause_mask)
        features['pause_count'] = len(pause_segments)
        features['total_pause_time'] = sum([seg[1] - seg[0] for seg in pause_segments])
        
        return features
    
    def _find_segments(self, mask):
        """Find contiguous True segments in boolean mask"""
        segments = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(mask)))
        
        return segments
    
    def create_feature_vector(self, normalized_strokes):
        """Create feature vector from multiple normalized strokes"""
        all_features = []
        
        for stroke in normalized_strokes:
            features = self.extract_normalized_features(stroke)
            all_features.append(features)
        
        # Aggregate features across strokes
        if not all_features:
            return {}
        
        aggregated = {}
        
        # Calculate statistics across strokes
        for key in all_features[0].keys():
            values = [f[key] for f in all_features if key in f]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_min'] = np.min(values)
        
        # Add stroke count
        aggregated['stroke_count'] = len(normalized_strokes)
        
        # Add total writing time
        if normalized_strokes and len(normalized_strokes[0]) > 0:
            start_time = normalized_strokes[0][0]['t']
            end_time = normalized_strokes[-1][-1]['t']
            aggregated['total_time'] = end_time - start_time
        
        return aggregated