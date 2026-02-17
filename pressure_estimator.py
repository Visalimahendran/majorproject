import numpy as np
from scipy import signal, stats, interpolate
import warnings
warnings.filterwarnings('ignore')

class PressureEstimator:
    """Estimate neural pressure from handwriting characteristics"""
    
    def __init__(self):
        # Pressure sensitivity parameters
        self.pressure_scaling_factors = {
            'velocity': 0.3,
            'acceleration': 0.2,
            'tremor': 0.25,
            'irregularity': 0.15,
            'hesitation': 0.1
        }
        
        # Neural Pressure Index (NPI) thresholds
        self.npi_thresholds = {
            'normal': 30,
            'mild_stress': 60,
            'severe_stress': 80
        }
    
    def estimate_pressure(self, stroke_data):
        """Estimate pressure from stroke data"""
        if not stroke_data or len(stroke_data) < 10:
            return {'npi': 0, 'pressure_level': 'insufficient_data'}
        
        # Extract features for pressure estimation
        features = self._extract_pressure_features(stroke_data)
        
        # Calculate Neural Pressure Index
        npi = self._calculate_npi(features)
        
        # Determine pressure level
        pressure_level = self._classify_pressure(npi)
        
        # Calculate writing force inference
        force_metrics = self._estimate_writing_force(features)
        
        return {
            'npi': float(npi),
            'pressure_level': pressure_level,
            'features': features,
            'force_metrics': force_metrics,
            'interpretation': self._interpret_pressure(npi, features)
        }
    
    def _extract_pressure_features(self, stroke_data):
        """Extract features relevant to pressure estimation"""
        features = {}
        
        # Extract basic stroke data
        if isinstance(stroke_data[0], dict):
            # Already processed stroke data
            x = np.array([p.get('x', 0) for p in stroke_data])
            y = np.array([p.get('y', 0) for p in stroke_data])
            t = np.array([p.get('timestamp', i) for i, p in enumerate(stroke_data)])
            pressures = np.array([p.get('pressure', 0.5) for p in stroke_data])
        else:
            # Raw stroke data
            x = np.array([p[0] for p in stroke_data])
            y = np.array([p[1] for p in stroke_data])
            t = np.arange(len(stroke_data))
            pressures = np.ones(len(stroke_data)) * 0.5
        
        # Convert timestamps if needed
        if hasattr(t[0], 'timestamp'):
            t = np.array([ts.timestamp() for ts in t])
        
        # Normalize time
        t = t - t[0]
        
        # 1. Velocity-based features
        velocity_features = self._calculate_velocity_features(x, y, t)
        features.update(velocity_features)
        
        # 2. Tremor analysis
        tremor_features = self._analyze_tremor(x, y, t)
        features.update(tremor_features)
        
        # 3. Irregularity metrics
        irregularity_features = self._calculate_irregularity(x, y, t)
        features.update(irregularity_features)
        
        # 4. Hesitation patterns
        hesitation_features = self._detect_hesitations(x, y, t, pressures)
        features.update(hesitation_features)
        
        # 5. Pressure dynamics (if pressure data available)
        if np.any(pressures > 0):
            pressure_features = self._analyze_pressure_dynamics(pressures, t)
            features.update(pressure_features)
        
        # 6. Spatial compression
        spatial_features = self._analyze_spatial_compression(x, y)
        features.update(spatial_features)
        
        return features
    
    def _calculate_velocity_features(self, x, y, t):
        """Calculate velocity-related features"""
        if len(t) < 2:
            return {}
        
        dt = np.diff(t)
        dt[dt == 0] = 1e-10
        
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        velocity = np.sqrt(vx**2 + vy**2)
        
        # Pad to match original length
        velocity = np.concatenate(([velocity[0]], velocity))
        
        # Jerk (rate of change of acceleration)
        if len(vx) > 1:
            dvx = np.diff(vx) / dt[:-1]
            dvy = np.diff(vy) / dt[:-1]
            acceleration = np.sqrt(dvx**2 + dvy**2)
            acceleration = np.concatenate(([acceleration[0]], acceleration))
            
            if len(acceleration) > 1:
                jerk = np.diff(acceleration) / dt[:-2] if len(dt) > 2 else np.array([0])
                jerk = np.concatenate(([jerk[0], jerk[0]], jerk))
            else:
                jerk = np.zeros_like(velocity)
        else:
            acceleration = np.zeros_like(velocity)
            jerk = np.zeros_like(velocity)
        
        features = {
            'velocity_mean': float(np.mean(velocity)),
            'velocity_std': float(np.std(velocity)),
            'velocity_cv': float(np.std(velocity) / (np.mean(velocity) + 1e-10)),
            'velocity_max': float(np.max(velocity)),
            'velocity_min': float(np.min(velocity)),
            'acceleration_mean': float(np.mean(acceleration)),
            'acceleration_std': float(np.std(acceleration)),
            'jerk_mean': float(np.mean(np.abs(jerk))),
            'velocity_skewness': float(stats.skew(velocity))
        }
        
        # Detect velocity spikes (indicative of tension)
        velocity_threshold = np.mean(velocity) + 2 * np.std(velocity)
        velocity_spikes = np.sum(velocity > velocity_threshold)
        features['velocity_spikes'] = int(velocity_spikes)
        features['velocity_spike_ratio'] = float(velocity_spikes / len(velocity))
        
        return features
    
    def _analyze_tremor(self, x, y, t, min_freq=3, max_freq=12):
        """Analyze tremor in handwriting"""
        if len(t) < 20:
            return {'tremor_power': 0, 'tremor_frequency': 0}
        
        # Calculate velocity for tremor analysis
        dt = np.diff(t)
        dt[dt == 0] = 1e-10
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        velocity = np.sqrt(vx**2 + vy**2)
        
        if len(velocity) < 10:
            return {'tremor_power': 0, 'tremor_frequency': 0}
        
        # Remove trend
        velocity_detrended = signal.detrend(velocity)
        
        # Calculate sampling frequency
        if np.mean(dt) > 0:
            fs = 1.0 / np.mean(dt)
        else:
            fs = 100  # Default
        
        # Calculate power spectrum
        freqs, power = signal.welch(velocity_detrended, fs=fs, nperseg=min(256, len(velocity)))
        
        # Focus on tremor frequency range (3-12 Hz)
        tremor_mask = (freqs >= min_freq) & (freqs <= max_freq)
        
        if np.any(tremor_mask):
            tremor_power = np.trapz(power[tremor_mask], freqs[tremor_mask])
            total_power = np.trapz(power, freqs)
            tremor_ratio = tremor_power / total_power if total_power > 0 else 0
            
            # Find dominant tremor frequency
            tremor_power_band = power[tremor_mask]
            tremor_freqs = freqs[tremor_mask]
            if len(tremor_power_band) > 0:
                dominant_idx = np.argmax(tremor_power_band)
                dominant_freq = tremor_freqs[dominant_idx]
            else:
                dominant_freq = 0
        else:
            tremor_power = 0
            tremor_ratio = 0
            dominant_freq = 0
        
        return {
            'tremor_power': float(tremor_power),
            'tremor_ratio': float(tremor_ratio),
            'tremor_frequency': float(dominant_freq),
            'tremor_intensity': float(tremor_ratio * 100)  # Scaled 0-100
        }
    
    def _calculate_irregularity(self, x, y, t):
        """Calculate handwriting irregularity metrics"""
        if len(x) < 10:
            return {'irregularity_index': 0}
        
        # 1. Path irregularity (deviation from smooth curve)
        # Fit a smooth spline
        try:
            t_norm = np.linspace(0, 1, len(x))
            spline_x = interpolate.UnivariateSpline(t_norm, x, k=3)
            spline_y = interpolate.UnivariateSpline(t_norm, y, k=3)
            
            x_smooth = spline_x(t_norm)
            y_smooth = spline_y(t_norm)
            
            # Calculate deviation
            deviation_x = x - x_smooth
            deviation_y = y - y_smooth
            total_deviation = np.sqrt(deviation_x**2 + deviation_y**2)
            
            irregularity_index = np.mean(total_deviation) / (np.std([x, y]) + 1e-10)
        except:
            irregularity_index = 0
        
        # 2. Jerk-cost metric (smoothness)
        if len(t) > 2:
            dt = np.diff(t)
            dt[dt == 0] = 1e-10
            
            # Calculate jerk (third derivative)
            dx = np.diff(x) / dt
            dy = np.diff(y) / dt
            
            if len(dx) > 1:
                ddx = np.diff(dx) / dt[:-1]
                ddy = np.diff(dy) / dt[:-1]
                
                if len(ddx) > 1:
                    dddx = np.diff(ddx) / dt[:-2]
                    dddy = np.diff(ddy) / dt[:-2]
                    jerk_magnitude = np.sqrt(dddx**2 + dddy**2)
                    jerk_cost = np.mean(jerk_magnitude**2)
                else:
                    jerk_cost = 0
            else:
                jerk_cost = 0
        else:
            jerk_cost = 0
        
        # 3. Fractal dimension (complexity)
        fd = self._estimate_fractal_dimension(x, y)
        
        return {
            'irregularity_index': float(irregularity_index),
            'jerk_cost': float(jerk_cost),
            'fractal_dimension': float(fd),
            'smoothness': float(1.0 / (1.0 + irregularity_index))
        }
    
    def _estimate_fractal_dimension(self, x, y):
        """Estimate fractal dimension using box-counting method"""
        if len(x) < 10:
            return 1.0
        
        # Normalize coordinates
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
        
        # Simple box-counting approximation
        n_points = len(x_norm)
        scales = [0.1, 0.05, 0.02, 0.01]
        counts = []
        
        for scale in scales:
            # Create grid
            x_bins = np.arange(0, 1 + scale, scale)
            y_bins = np.arange(0, 1 + scale, scale)
            
            # Count occupied boxes
            occupied = np.zeros((len(x_bins)-1, len(y_bins)-1), dtype=bool)
            
            for xi, yi in zip(x_norm, y_norm):
                x_idx = int(xi / scale)
                y_idx = int(yi / scale)
                if x_idx < occupied.shape[0] and y_idx < occupied.shape[1]:
                    occupied[x_idx, y_idx] = True
            
            counts.append(np.sum(occupied))
        
        # Linear fit in log-log space
        if len(counts) >= 2:
            log_scales = np.log(1 / np.array(scales[:len(counts)]))
            log_counts = np.log(np.array(counts))
            
            try:
                coeffs = np.polyfit(log_scales, log_counts, 1)
                fd = coeffs[0]  # Slope is fractal dimension estimate
            except:
                fd = 1.0
        else:
            fd = 1.0
        
        return max(1.0, min(2.0, fd))
    
    def _detect_hesitations(self, x, y, t, pressures):
        """Detect hesitation patterns in handwriting"""
        if len(t) < 10:
            return {'hesitation_count': 0, 'pause_ratio': 0}
        
        # Calculate instantaneous speed
        dt = np.diff(t)
        dt[dt == 0] = 1e-10
        dx = np.diff(x) / dt
        dy = np.diff(y) / dt
        speed = np.sqrt(dx**2 + dy**2)
        speed = np.concatenate(([speed[0]], speed))
        
        # Define hesitation threshold (low speed)
        speed_threshold = np.percentile(speed, 25)  # Bottom 25%
        
        # Find hesitation segments
        hesitation_mask = speed < speed_threshold
        hesitation_segments = self._find_contiguous_segments(hesitation_mask)
        
        # Filter very short hesitations (< 0.05s)
        valid_segments = []
        for start, end in hesitation_segments:
            duration = t[min(end, len(t)-1)] - t[start]
            if duration > 0.05:  # 50ms minimum
                valid_segments.append((start, end))
        
        hesitation_count = len(valid_segments)
        
        # Calculate total hesitation time
        hesitation_time = 0
        for start, end in valid_segments:
            hesitation_time += t[min(end, len(t)-1)] - t[start]
        
        total_time = t[-1] - t[0]
        pause_ratio = hesitation_time / total_time if total_time > 0 else 0
        
        # Analyze hesitation patterns
        hesitation_durations = []
        for start, end in valid_segments:
            duration = t[min(end, len(t)-1)] - t[start]
            hesitation_durations.append(duration)
        
        if hesitation_durations:
            hesitation_stats = {
                'mean_duration': float(np.mean(hesitation_durations)),
                'std_duration': float(np.std(hesitation_durations)),
                'max_duration': float(np.max(hesitation_durations))
            }
        else:
            hesitation_stats = {}
        
        # Pressure changes during hesitations
        pressure_changes = []
        if len(pressures) == len(t) and hesitation_count > 0:
            for start, end in valid_segments:
                if end < len(pressures):
                    pressure_before = pressures[max(0, start-2):start+1]
                    pressure_during = pressures[start:min(end+1, len(pressures))]
                    
                    if len(pressure_before) > 0 and len(pressure_during) > 0:
                        change = np.mean(pressure_during) - np.mean(pressure_before)
                        pressure_changes.append(change)
        
        return {
            'hesitation_count': hesitation_count,
            'hesitation_ratio': float(pause_ratio),
            'hesitation_time': float(hesitation_time),
            **hesitation_stats,
            'pressure_change_during_hesitation': float(np.mean(pressure_changes)) if pressure_changes else 0
        }
    
    def _find_contiguous_segments(self, mask):
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
    
    def _analyze_pressure_dynamics(self, pressures, t):
        """Analyze pressure dynamics if pressure data is available"""
        if len(pressures) < 10:
            return {}
        
        features = {
            'pressure_mean': float(np.mean(pressures)),
            'pressure_std': float(np.std(pressures)),
            'pressure_cv': float(np.std(pressures) / (np.mean(pressures) + 1e-10)),
            'pressure_max': float(np.max(pressures)),
            'pressure_min': float(np.min(pressures)),
            'pressure_range': float(np.max(pressures) - np.min(pressures))
        }
        
        # Pressure variability over time
        if len(t) > 1:
            dt = np.diff(t)
            dt[dt == 0] = 1e-10
            pressure_gradient = np.diff(pressures) / dt
            features['pressure_gradient_mean'] = float(np.mean(np.abs(pressure_gradient)))
            features['pressure_gradient_std'] = float(np.std(pressure_gradient))
        
        # Detect pressure spikes (excessive force)
        pressure_threshold = np.mean(pressures) + 2 * np.std(pressures)
        pressure_spikes = np.sum(pressures > pressure_threshold)
        features['pressure_spikes'] = int(pressure_spikes)
        features['pressure_spike_ratio'] = float(pressure_spikes / len(pressures))
        
        # Pressure rhythm (autocorrelation)
        if len(pressures) > 20:
            pressure_norm = pressures - np.mean(pressures)
            autocorr = np.correlate(pressure_norm, pressure_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            
            # Find correlation length
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                features['pressure_correlation_length'] = float(zero_crossings[0])
            else:
                features['pressure_correlation_length'] = float(len(autocorr))
        
        return features
    
    def _analyze_spatial_compression(self, x, y):
        """Analyze spatial compression (tight vs loose writing)"""
        if len(x) < 10:
            return {'spatial_density': 0}
        
        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        try:
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            hull_area = hull.volume if hasattr(hull, 'volume') else hull.area
        except:
            # Fallback to bounding box area
            hull_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
        
        # Total path length
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        # Spatial density (path length per unit area)
        spatial_density = path_length / (hull_area + 1e-10)
        
        # Characteristic writing size
        writing_size = np.sqrt(hull_area)
        
        return {
            'spatial_density': float(spatial_density),
            'writing_size': float(writing_size),
            'hull_area': float(hull_area),
            'path_length': float(path_length),
            'compression_ratio': float(1.0 / (1.0 + spatial_density))  # Inverse of density
        }
    
    def _calculate_npi(self, features):
        """Calculate Neural Pressure Index (NPI)"""
        npi = 0
        
        # Weighted combination of features
        weights = self.pressure_scaling_factors
        
        # Velocity component (higher variability = more pressure)
        if 'velocity_cv' in features:
            npi += weights['velocity'] * min(100, features['velocity_cv'] * 100)
        
        # Acceleration component (jerkiness)
        if 'jerk_mean' in features:
            npi += weights['acceleration'] * min(100, features['jerk_mean'] * 1000)
        
        # Tremor component
        if 'tremor_intensity' in features:
            npi += weights['tremor'] * features['tremor_intensity']
        
        # Irregularity component
        if 'irregularity_index' in features:
            npi += weights['irregularity'] * min(100, features['irregularity_index'] * 100)
        
        # Hesitation component
        if 'hesitation_ratio' in features:
            npi += weights['hesitation'] * min(100, features['hesitation_ratio'] * 200)
        
        # Pressure spikes (if available)
        if 'pressure_spike_ratio' in features:
            npi += 0.1 * min(100, features['pressure_spike_ratio'] * 200)
        
        # Clip to 0-100 range
        npi = max(0, min(100, npi))
        
        return npi
    
    def _classify_pressure(self, npi):
        """Classify pressure level based on NPI"""
        if npi < self.npi_thresholds['normal']:
            return 'normal'
        elif npi < self.npi_thresholds['mild_stress']:
            return 'mild_stress'
        elif npi < self.npi_thresholds['severe_stress']:
            return 'moderate_stress'
        else:
            return 'severe_stress'
    
    def _estimate_writing_force(self, features):
        """Estimate writing force from features"""
        force_metrics = {}
        
        # Base force estimation (normalized 0-1)
        base_force = 0.5
        
        # Adjust based on pressure features
        if 'pressure_mean' in features:
            base_force = min(1.0, features['pressure_mean'] / 0.8)  # Assuming 0.8 is max
        
        # Adjust based on spatial compression
        if 'spatial_density' in features:
            density_factor = min(1.0, features['spatial_density'] / 10)  # Normalize
            base_force = base_force * (1 + 0.3 * density_factor)
        
        # Adjust based on jerkiness
        if 'jerk_mean' in features:
            jerk_factor = min(1.0, features['jerk_mean'] / 5)  # Normalize
            base_force = base_force * (1 + 0.2 * jerk_factor)
        
        force_metrics['estimated_force'] = float(min(1.0, base_force))
        force_metrics['force_level'] = self._classify_force(force_metrics['estimated_force'])
        
        # Additional force metrics
        force_metrics['force_variability'] = features.get('pressure_cv', 0.3) if 'pressure_cv' in features else 0.3
        force_metrics['force_consistency'] = 1.0 / (1.0 + force_metrics['force_variability'])
        
        return force_metrics
    
    def _classify_force(self, force_value):
        """Classify writing force level"""
        if force_value < 0.3:
            return 'light'
        elif force_value < 0.6:
            return 'medium'
        elif force_value < 0.8:
            return 'firm'
        else:
            return 'heavy'
    
    def _interpret_pressure(self, npi, features):
        """Generate human-readable interpretation of pressure results"""
        interpretations = []
        
        # NPI-based interpretation
        if npi < 30:
            interpretations.append("Normal neural-motor pressure. Writing appears relaxed and controlled.")
        elif npi < 60:
            interpretations.append("Mild elevation in neural-motor pressure. Some signs of tension detected.")
        elif npi < 80:
            interpretations.append("Moderate neural-motor pressure. Noticeable tension and irregularity.")
        else:
            interpretations.append("High neural-motor pressure. Significant tension, possibly indicating stress.")
        
        # Feature-specific interpretations
        if features.get('tremor_intensity', 0) > 50:
            interpretations.append("Elevated tremor detected, which can indicate nervous system arousal.")
        
        if features.get('hesitation_ratio', 0) > 0.1:
            interpretations.append("Frequent hesitations suggest cognitive load or uncertainty.")
        
        if features.get('irregularity_index', 0) > 0.5:
            interpretations.append("High irregularity in stroke patterns.")
        
        if features.get('velocity_spike_ratio', 0) > 0.1:
            interpretations.append("Inconsistent velocity with sudden changes.")
        
        # Recommendations based on findings
        recommendations = []
        if npi > 60:
            recommendations.append("Consider relaxation exercises before writing tasks.")
            recommendations.append("Practice slow, deliberate writing to improve control.")
            recommendations.append("Ensure comfortable writing position and grip.")
        
        if features.get('tremor_intensity', 0) > 50:
            recommendations.append("Hand stability exercises may help reduce tremor.")
        
        return {
            'summary': ' '.join(interpretations),
            'recommendations': recommendations,
            'key_indicators': {
                'tremor': features.get('tremor_intensity', 0),
                'hesitation': features.get('hesitation_ratio', 0),
                'irregularity': features.get('irregularity_index', 0),
                'force_consistency': features.get('force_consistency', 0.5) if 'force_consistency' in features else 0.5
            }
        }
    
    def batch_analyze(self, stroke_data_list):
        """Analyze multiple stroke datasets"""
        results = []
        
        for stroke_data in stroke_data_list:
            result = self.estimate_pressure(stroke_data)
            results.append(result)
        
        # Aggregate results
        if results:
            npis = [r['npi'] for r in results]
            aggregate = {
                'mean_npi': float(np.mean(npis)),
                'std_npi': float(np.std(npis)),
                'min_npi': float(np.min(npis)),
                'max_npi': float(np.max(npis)),
                'trend': self._analyze_trend(npis),
                'individual_results': results
            }
            return aggregate
        else:
            return {'error': 'No valid stroke data provided'}
    
    def _analyze_trend(self, npis):
        """Analyze trend in NPI values over time"""
        if len(npis) < 3:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(npis))
        slope, intercept = np.polyfit(x, npis, 1)
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'