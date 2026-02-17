import numpy as np
from scipy import signal, stats, fft
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

class TemporalAnalyzer:
    """Analyze temporal patterns in handwriting"""
    
    def __init__(self, window_size=50, overlap=0.5, sampling_rate=100):
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        
    def analyze(self, strokes):
        """Analyze temporal patterns across strokes"""
        if not strokes:
            return {}
        
        # Combine all strokes into continuous data
        combined_data = self._combine_strokes(strokes)
        
        # Extract temporal features
        features = {}
        
        # 1. Timing features
        features.update(self._extract_timing_features(strokes))
        
        # 2. Rhythm and regularity
        features.update(self._analyze_rhythm(combined_data))
        
        # 3. Pause analysis
        features.update(self._analyze_pauses(combined_data))
        
        # 4. Velocity profile analysis
        features.update(self._analyze_velocity_profile(combined_data))
        
        # 5. Frequency domain analysis
        features.update(self._frequency_domain_analysis(combined_data))
        
        # 6. Entropy and complexity
        features.update(self._calculate_entropy(combined_data))
        
        # 7. Statistical moments
        features.update(self._calculate_statistical_moments(combined_data))
        
        return features
    
    def _combine_strokes(self, strokes):
        """Combine multiple strokes into continuous data"""
        combined = {
            'x': [],
            'y': [],
            'time': [],
            'pressure': [],
            'velocity': [],
            'acceleration': []
        }
        
        current_time = 0
        
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            # Extract data from stroke
            x = [p['x'] for p in stroke]
            y = [p['y'] for p in stroke]
            
            # Handle timestamps
            if 'timestamp' in stroke[0]:
                timestamps = [p['timestamp'] for p in stroke]
                if hasattr(timestamps[0], 'timestamp'):
                    timestamps = [ts.timestamp() for ts in timestamps]
                times = np.array(timestamps) - timestamps[0] + current_time
            else:
                # Create synthetic timestamps
                times = np.arange(len(stroke)) / self.sampling_rate + current_time
            
            # Calculate velocity and acceleration
            if len(stroke) >= 3:
                velocity, acceleration = self._calculate_kinematics(x, y, times)
            else:
                velocity = np.zeros(len(stroke))
                acceleration = np.zeros(len(stroke))
            
            # Extract pressure if available
            if 'pressure' in stroke[0]:
                pressure = [p['pressure'] for p in stroke]
            else:
                pressure = [0.5] * len(stroke)
            
            # Append to combined data
            combined['x'].extend(x)
            combined['y'].extend(y)
            combined['time'].extend(times.tolist())
            combined['pressure'].extend(pressure)
            combined['velocity'].extend(velocity.tolist())
            combined['acceleration'].extend(acceleration.tolist())
            
            # Update current time for next stroke
            current_time = times[-1] + 0.1  # Add small gap
        
        # Convert to numpy arrays
        for key in combined:
            combined[key] = np.array(combined[key])
        
        return combined
    
    def _calculate_kinematics(self, x, y, t):
        """Calculate velocity and acceleration"""
        dt = np.diff(t)
        dt[dt == 0] = 1e-10
        
        # Velocity
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        velocity = np.sqrt(vx**2 + vy**2)
        
        # Pad to original length
        velocity = np.concatenate(([velocity[0]], velocity))
        
        # Acceleration
        if len(vx) > 1:
            dvx = np.diff(vx) / dt[:-1]
            dvy = np.diff(vy) / dt[:-1]
            acceleration = np.sqrt(dvx**2 + dvy**2)
            acceleration = np.concatenate(([acceleration[0], acceleration[0]], acceleration))
        else:
            acceleration = np.zeros_like(velocity)
        
        return velocity, acceleration
    
    def _extract_timing_features(self, strokes):
        """Extract timing-related features"""
        if not strokes:
            return {}
        
        # Calculate stroke durations
        stroke_durations = []
        inter_stroke_intervals = []
        
        for i, stroke in enumerate(strokes):
            if len(stroke) < 2:
                continue
            
            # Stroke duration
            if 'timestamp' in stroke[0]:
                start = stroke[0]['timestamp']
                end = stroke[-1]['timestamp']
                if hasattr(start, 'timestamp'):
                    start = start.timestamp()
                    end = end.timestamp()
                duration = end - start
            else:
                duration = len(stroke) / self.sampling_rate
            
            stroke_durations.append(duration)
            
            # Inter-stroke interval (to next stroke)
            if i < len(strokes) - 1:
                next_stroke = strokes[i + 1]
                if len(next_stroke) > 0 and 'timestamp' in next_stroke[0]:
                    next_start = next_stroke[0]['timestamp']
                    if hasattr(next_start, 'timestamp'):
                        next_start = next_start.timestamp()
                    interval = next_start - end
                    inter_stroke_intervals.append(interval)
        
        features = {}
        
        if stroke_durations:
            features['stroke_duration_mean'] = float(np.mean(stroke_durations))
            features['stroke_duration_std'] = float(np.std(stroke_durations))
            features['stroke_duration_cv'] = float(features['stroke_duration_std'] / 
                                                  (features['stroke_duration_mean'] + 1e-10))
            features['stroke_duration_min'] = float(np.min(stroke_durations))
            features['stroke_duration_max'] = float(np.max(stroke_durations))
        
        if inter_stroke_intervals:
            features['inter_stroke_interval_mean'] = float(np.mean(inter_stroke_intervals))
            features['inter_stroke_interval_std'] = float(np.std(inter_stroke_intervals))
            features['inter_stroke_interval_cv'] = float(features['inter_stroke_interval_std'] / 
                                                        (features['inter_stroke_interval_mean'] + 1e-10))
        
        # Total writing time
        if strokes and 'timestamp' in strokes[0][0]:
            first_start = strokes[0][0]['timestamp']
            last_end = strokes[-1][-1]['timestamp']
            if hasattr(first_start, 'timestamp'):
                first_start = first_start.timestamp()
                last_end = last_end.timestamp()
            features['total_writing_time'] = float(last_end - first_start)
        
        # Stroke rate (strokes per second)
        if features.get('total_writing_time', 0) > 0:
            features['stroke_rate'] = len(strokes) / features['total_writing_time']
        
        return features
    
    def _analyze_rhythm(self, combined_data):
        """Analyze writing rhythm and regularity"""
        if len(combined_data['time']) < 10:
            return {}
        
        time = combined_data['time']
        velocity = combined_data['velocity']
        
        # Find peaks in velocity (stroke movements)
        if len(velocity) > 10:
            peaks, properties = signal.find_peaks(velocity, 
                                                 height=np.mean(velocity) + np.std(velocity),
                                                 distance=5)
            
            if len(peaks) > 3:
                # Calculate inter-peak intervals
                peak_times = time[peaks]
                inter_peak_intervals = np.diff(peak_times)
                
                # Rhythm regularity metrics
                cv_intervals = np.std(inter_peak_intervals) / (np.mean(inter_peak_intervals) + 1e-10)
                
                # Autocorrelation of intervals
                if len(inter_peak_intervals) > 10:
                    autocorr = np.correlate(inter_peak_intervals - np.mean(inter_peak_intervals),
                                           inter_peak_intervals - np.mean(inter_peak_intervals),
                                           mode='full')
                    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
                    
                    # Find first zero crossing
                    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                    if len(zero_crossings) > 0:
                        correlation_length = zero_crossings[0]
                    else:
                        correlation_length = len(autocorr)
                else:
                    correlation_length = 0
                
                return {
                    'rhythm_peak_count': len(peaks),
                    'rhythm_mean_interval': float(np.mean(inter_peak_intervals)),
                    'rhythm_interval_cv': float(cv_intervals),
                    'rhythm_autocorr_length': float(correlation_length),
                    'rhythm_regularity': float(1.0 / (1.0 + cv_intervals))
                }
        
        return {}
    
    def _analyze_pauses(self, combined_data):
        """Analyze pause patterns during writing"""
        if len(combined_data['velocity']) < 10:
            return {}
        
        velocity = combined_data['velocity']
        time = combined_data['time']
        
        # Define pause threshold (velocity below threshold)
        velocity_threshold = np.percentile(velocity, 20)  # Bottom 20% as pause
        
        # Find pause segments
        pause_mask = velocity < velocity_threshold
        pause_segments = self._find_contiguous_segments(pause_mask)
        
        if not pause_segments:
            return {}
        
        # Calculate pause statistics
        pause_durations = []
        pause_velocities = []
        
        for start, end in pause_segments:
            if end - start > 1:  # Minimum 2 points for a pause
                duration = time[end] - time[start]
                pause_durations.append(duration)
                pause_vel = np.mean(velocity[start:end])
                pause_velocities.append(pause_vel)
        
        if not pause_durations:
            return {}
        
        features = {
            'pause_count': len(pause_durations),
            'pause_total_duration': float(np.sum(pause_durations)),
            'pause_mean_duration': float(np.mean(pause_durations)),
            'pause_std_duration': float(np.std(pause_durations)),
            'pause_max_duration': float(np.max(pause_durations)),
            'pause_min_duration': float(np.min(pause_durations)),
            'pause_fraction': float(np.sum(pause_durations) / (time[-1] - time[0] + 1e-10)),
            'pause_mean_velocity': float(np.mean(pause_velocities))
        }
        
        # Analyze pause distribution
        if len(pause_durations) > 5:
            # Fit exponential distribution to pause durations
            try:
                loc, scale = stats.expon.fit(pause_durations)
                features['pause_expon_loc'] = float(loc)
                features['pause_expon_scale'] = float(scale)
                
                # KS test for exponential distribution
                ks_stat, ks_p = stats.kstest(pause_durations, 'expon', args=(loc, scale))
                features['pause_ks_statistic'] = float(ks_stat)
                features['pause_ks_pvalue'] = float(ks_p)
            except:
                pass
        
        # Detect micro-pauses (very short pauses that might indicate hesitation)
        micro_pause_threshold = 0.1  # 100ms
        micro_pauses = [d for d in pause_durations if d < micro_pause_threshold]
        
        features['micro_pause_count'] = len(micro_pauses)
        features['micro_pause_fraction'] = len(micro_pauses) / (len(pause_durations) + 1e-10)
        
        return features
    
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
    
    def _analyze_velocity_profile(self, combined_data):
        """Analyze velocity profile patterns"""
        if len(combined_data['velocity']) < 20:
            return {}
        
        velocity = combined_data['velocity']
        
        # Calculate velocity statistics
        vel_mean = np.mean(velocity)
        vel_std = np.std(velocity)
        
        # Skewness and kurtosis of velocity distribution
        vel_skew = stats.skew(velocity)
        vel_kurtosis = stats.kurtosis(velocity)
        
        # Velocity autocorrelation
        if len(velocity) > 50:
            vel_norm = velocity - vel_mean
            autocorr = np.correlate(vel_norm, vel_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            
            # Find correlation time (first zero crossing)
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                correlation_time = zero_crossings[0] / self.sampling_rate
            else:
                correlation_time = len(autocorr) / self.sampling_rate
        else:
            correlation_time = 0
        
        # Detect velocity peaks and valleys
        peaks, _ = signal.find_peaks(velocity, distance=10)
        valleys, _ = signal.find_peaks(-velocity, distance=10)
        
        # Calculate modulation depth
        if len(peaks) > 0 and len(valleys) > 0:
            peak_vals = velocity[peaks]
            valley_vals = velocity[valleys]
            modulation_depth = np.mean(peak_vals) - np.mean(valley_vals)
        else:
            modulation_depth = 0
        
        features = {
            'velocity_mean': float(vel_mean),
            'velocity_std': float(vel_std),
            'velocity_cv': float(vel_std / (vel_mean + 1e-10)),
            'velocity_skewness': float(vel_skew),
            'velocity_kurtosis': float(vel_kurtosis),
            'velocity_autocorr_time': float(correlation_time),
            'velocity_peak_count': len(peaks),
            'velocity_valley_count': len(valleys),
            'velocity_modulation_depth': float(modulation_depth)
        }
        
        # Calculate velocity gradient (change in velocity over time)
        if len(velocity) > 10:
            vel_gradient = np.gradient(velocity)
            features['velocity_gradient_mean'] = float(np.mean(np.abs(vel_gradient)))
            features['velocity_gradient_std'] = float(np.std(vel_gradient))
        
        return features
    
    def _frequency_domain_analysis(self, combined_data):
        """Analyze frequency domain characteristics"""
        if len(combined_data['velocity']) < 100:
            return {}
        
        velocity = combined_data['velocity']
        
        # Remove DC component and detrend
        vel_detrended = signal.detrend(velocity)
        
        # Apply windowing function
        window = signal.windows.hann(len(vel_detrended))
        vel_windowed = vel_detrended * window
        
        # Calculate FFT
        fft_values = fft.fft(vel_windowed)
        freqs = fft.fftfreq(len(vel_windowed), 1/self.sampling_rate)
        
        # Get positive frequencies only
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        fft_pos = fft_values[pos_mask]
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_pos)**2
        
        # Find dominant frequency
        if len(power_spectrum) > 0:
            dominant_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC
            dominant_freq = freqs_pos[dominant_idx]
            dominant_power = power_spectrum[dominant_idx]
        else:
            dominant_freq = 0
            dominant_power = 0
        
        # Calculate spectral features
        total_power = np.sum(power_spectrum)
        
        # Power in different frequency bands (Hz)
        tremor_band = (3, 12)  # Physiological tremor
        voluntary_band = (0.5, 3)  # Voluntary movements
        
        tremor_mask = (freqs_pos >= tremor_band[0]) & (freqs_pos <= tremor_band[1])
        voluntary_mask = (freqs_pos >= voluntary_band[0]) & (freqs_pos <= voluntary_band[1])
        
        tremor_power = np.sum(power_spectrum[tremor_mask]) if np.any(tremor_mask) else 0
        voluntary_power = np.sum(power_spectrum[voluntary_mask]) if np.any(voluntary_mask) else 0
        
        # Spectral centroid (average frequency weighted by power)
        if total_power > 0:
            spectral_centroid = np.sum(freqs_pos * power_spectrum) / total_power
            spectral_spread = np.sqrt(np.sum((freqs_pos - spectral_centroid)**2 * power_spectrum) / total_power)
            spectral_skewness = (np.sum((freqs_pos - spectral_centroid)**3 * power_spectrum) / 
                               (total_power * spectral_spread**3))
            spectral_kurtosis = (np.sum((freqs_pos - spectral_centroid)**4 * power_spectrum) / 
                               (total_power * spectral_spread**4)) - 3
        else:
            spectral_centroid = 0
            spectral_spread = 0
            spectral_skewness = 0
            spectral_kurtosis = 0
        
        features = {
            'spectral_dominant_freq': float(dominant_freq),
            'spectral_dominant_power': float(dominant_power),
            'spectral_total_power': float(total_power),
            'spectral_tremor_power': float(tremor_power),
            'spectral_voluntary_power': float(voluntary_power),
            'spectral_tremor_ratio': float(tremor_power / (total_power + 1e-10)),
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_skewness': float(spectral_skewness),
            'spectral_kurtosis': float(spectral_kurtosis)
        }
        
        # Calculate spectral entropy
        if total_power > 0:
            # Normalize power spectrum to create probability distribution
            power_norm = power_spectrum / total_power
            # Remove zeros for log calculation
            power_norm = power_norm[power_norm > 0]
            spectral_entropy = -np.sum(power_norm * np.log2(power_norm))
            features['spectral_entropy'] = float(spectral_entropy)
        
        return features
    
    def _calculate_entropy(self, combined_data):
        """Calculate entropy measures of the signal"""
        if len(combined_data['velocity']) < 50:
            return {}
        
        velocity = combined_data['velocity']
        
        # Sample entropy
        sampen = self._calculate_sample_entropy(velocity, m=2, r=0.2*np.std(velocity))
        
        # Approximate entropy
        apen = self._calculate_approximate_entropy(velocity, m=2, r=0.2*np.std(velocity))
        
        # Multiscale entropy (simplified)
        mse = self._calculate_multiscale_entropy(velocity, scale=3)
        
        features = {
            'entropy_sample': float(sampen),
            'entropy_approximate': float(apen),
            'entropy_multiscale': float(mse)
        }
        
        return features
    
    def _calculate_sample_entropy(self, data, m=2, r=None):
        """Calculate Sample Entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        # Split data into templates
        templates = np.array([data[i:i+m] for i in range(N - m)])
        next_templates = np.array([data[i:i+m+1] for i in range(N - m - 1)])
        
        # Count matches
        B = 0
        A = 0
        
        for i in range(len(templates) - 1):
            # Distance between templates
            dist = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            B += np.sum(dist <= r)
            
            if i < len(next_templates) - 1:
                # Distance between extended templates
                dist_ext = np.max(np.abs(next_templates[i+1:] - next_templates[i]), axis=1)
                A += np.sum(dist_ext <= r)
        
        if B == 0 or A == 0:
            return 0
        
        return -np.log(A / B)
    
    def _calculate_approximate_entropy(self, data, m=2, r=None):
        """Calculate Approximate Entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        def _phi(m):
            # Create templates
            templates = np.array([data[i:i+m] for i in range(N - m + 1)])
            
            # Count matches
            C = np.zeros(len(templates))
            for i in range(len(templates)):
                dist = np.max(np.abs(templates - templates[i]), axis=1)
                C[i] = np.sum(dist <= r) / (N - m + 1)
            
            return np.mean(np.log(C))
        
        return _phi(m) - _phi(m + 1)
    
    def _calculate_multiscale_entropy(self, data, scale=3):
        """Calculate Multiscale Entropy (simplified)"""
        if len(data) < scale * 10:
            return 0
        
        # Coarse-grain the data
        coarse_data = []
        for i in range(0, len(data) - scale + 1, scale):
            coarse_data.append(np.mean(data[i:i+scale]))
        
        # Calculate sample entropy of coarse-grained data
        return self._calculate_sample_entropy(np.array(coarse_data))
    
    def _calculate_statistical_moments(self, combined_data):
        """Calculate statistical moments of various signals"""
        features = {}
        
        signals = {
            'velocity': combined_data['velocity'],
            'acceleration': combined_data['acceleration'],
            'pressure': combined_data['pressure']
        }
        
        for name, signal_data in signals.items():
            if len(signal_data) > 10:
                features[f'{name}_mean'] = float(np.mean(signal_data))
                features[f'{name}_std'] = float(np.std(signal_data))
                features[f'{name}_skewness'] = float(stats.skew(signal_data))
                features[f'{name}_kurtosis'] = float(stats.kurtosis(signal_data))
                features[f'{name}_range'] = float(np.max(signal_data) - np.min(signal_data))
                
                # Quartiles
                q1, q2, q3 = np.percentile(signal_data, [25, 50, 75])
                features[f'{name}_q1'] = float(q1)
                features[f'{name}_median'] = float(q2)
                features[f'{name}_q3'] = float(q3)
                features[f'{name}_iqr'] = float(q3 - q1)
        
        return features
    
    def sliding_window_analysis(self, combined_data):
        """Perform sliding window analysis for local temporal patterns"""
        if len(combined_data['time']) < self.window_size * 2:
            return []
        
        time = combined_data['time']
        velocity = combined_data['velocity']
        
        window_results = []
        step = int(self.window_size * (1 - self.overlap))
        
        for i in range(0, len(velocity) - self.window_size + 1, step):
            window_start = i
            window_end = i + self.window_size
            
            window_data = {
                'time': time[window_start:window_end],
                'velocity': velocity[window_start:window_end]
            }
            
            # Analyze this window
            window_features = self._analyze_velocity_profile({'velocity': window_data['velocity']})
            window_features['window_start'] = float(time[window_start])
            window_features['window_end'] = float(time[window_end])
            window_features['window_duration'] = float(time[window_end] - time[window_start])
            
            window_results.append(window_features)
        
        return window_results
    
    def detect_writing_style_changes(self, combined_data):
        """Detect changes in writing style over time"""
        window_results = self.sliding_window_analysis(combined_data)
        
        if len(window_results) < 3:
            return []
        
        # Track changes in key features
        change_points = []
        
        key_features = ['velocity_mean', 'velocity_std', 'velocity_skewness']
        
        for feature in key_features:
            values = [w.get(feature, 0) for w in window_results]
            
            if len(values) > 10:
                # Use change point detection
                try:
                    # Simple threshold-based detection
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    for i in range(1, len(values)):
                        change = abs(values[i] - values[i-1])
                        if change > 2 * std_val:
                            change_points.append({
                                'feature': feature,
                                'window_index': i,
                                'time': window_results[i]['window_start'],
                                'change_magnitude': change
                            })
                except:
                    pass
        
        # Remove duplicates (similar time points)
        if change_points:
            change_points.sort(key=lambda x: x['time'])
            
            # Cluster nearby change points
            clustered = []
            current_cluster = []
            time_threshold = 1.0  # 1 second
            
            for cp in change_points:
                if not current_cluster or (cp['time'] - current_cluster[-1]['time'] <= time_threshold):
                    current_cluster.append(cp)
                else:
                    if current_cluster:
                        # Take the strongest change in cluster
                        strongest = max(current_cluster, key=lambda x: x['change_magnitude'])
                        clustered.append(strongest)
                    current_cluster = [cp]
            
            if current_cluster:
                strongest = max(current_cluster, key=lambda x: x['change_magnitude'])
                clustered.append(strongest)
            
            change_points = clustered
        
        return change_points