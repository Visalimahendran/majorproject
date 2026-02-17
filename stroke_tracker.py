import numpy as np
from scipy import spatial, signal
import warnings
warnings.filterwarnings('ignore')

class StrokeTracker:
    """Track and analyze handwriting strokes"""
    
    def __init__(self, min_points=10, max_gap=50, smooth_window=5):
        self.min_points = min_points
        self.max_gap = max_gap
        self.smooth_window = smooth_window
        
    def track(self, stroke_data):
        """Track strokes from raw point data"""
        if not stroke_data:
            return []
        
        # Group points into strokes based on gaps
        strokes = self._segment_strokes(stroke_data)
        
        # Filter short strokes
        strokes = [s for s in strokes if len(s) >= self.min_points]
        
        # Smooth strokes
        smoothed_strokes = []
        for stroke in strokes:
            smoothed = self._smooth_stroke(stroke)
            if len(smoothed) >= self.min_points:
                smoothed_strokes.append(smoothed)
        
        return smoothed_strokes
    
    def _segment_strokes(self, points):
        """Segment points into individual strokes"""
        if len(points) < 2:
            return [points] if points else []
        
        strokes = []
        current_stroke = [points[0]]
        
        for i in range(1, len(points)):
            current_point = points[i]
            last_point = current_stroke[-1]
            
            # Calculate distance between points
            distance = self._calculate_distance(current_point, last_point)
            
            # Check if pen was lifted (large gap)
            if distance > self.max_gap and len(current_stroke) >= self.min_points:
                # End current stroke and start new one
                strokes.append(current_stroke)
                current_stroke = [current_point]
            else:
                # Continue current stroke
                current_stroke.append(current_point)
        
        # Add last stroke
        if len(current_stroke) >= self.min_points:
            strokes.append(current_stroke)
        
        return strokes
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        if isinstance(p1, dict) and isinstance(p2, dict):
            # Points are dictionaries with x, y coordinates
            dx = p1.get('x', 0) - p2.get('x', 0)
            dy = p1.get('y', 0) - p2.get('y', 0)
        elif isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list)):
            # Points are tuples/lists
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
        else:
            return float('inf')
        
        return np.sqrt(dx**2 + dy**2)
    
    def _smooth_stroke(self, stroke):
        """Apply smoothing to stroke points"""
        if len(stroke) < self.smooth_window:
            return stroke
        
        # Extract coordinates
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        
        # Apply Savitzky-Golay filter for smoothness
        try:
            window_length = min(self.smooth_window, len(x) - 1)
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            
            if window_length >= 3:
                x_smooth = signal.savgol_filter(x, window_length, 2)
                y_smooth = signal.savgol_filter(y, window_length, 2)
            else:
                x_smooth, y_smooth = x, y
        except:
            # Fallback to moving average
            x_smooth = np.convolve(x, np.ones(self.smooth_window)/self.smooth_window, mode='same')
            y_smooth = np.convolve(y, np.ones(self.smooth_window)/self.smooth_window, mode='same')
        
        # Create smoothed stroke
        smoothed_stroke = []
        for i, point in enumerate(stroke):
            smoothed_point = point.copy()
            smoothed_point['x'] = float(x_smooth[i])
            smoothed_point['y'] = float(y_smooth[i])
            smoothed_stroke.append(smoothed_point)
        
        return smoothed_stroke
    
    def analyze_stroke_kinematics(self, stroke):
        """Analyze stroke kinematics (velocity, acceleration, jerk)"""
        if len(stroke) < 3:
            return {}
        
        # Extract data
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        t = np.array([p.get('timestamp', i) for i, p in enumerate(stroke)])
        
        # Convert timestamps if needed
        if hasattr(t[0], 'timestamp'):
            t = np.array([ts.timestamp() for ts in t])
        
        # Normalize time to start at 0
        t = t - t[0]
        
        # Calculate first derivative (velocity)
        dt = np.diff(t)
        dt[dt == 0] = 1e-10  # Avoid division by zero
        
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Pad to original length
        vx = np.concatenate(([vx[0]], vx))
        vy = np.concatenate(([vy[0]], vy))
        
        speed = np.sqrt(vx**2 + vy**2)
        
        # Calculate second derivative (acceleration)
        dvx = np.diff(vx) / dt[:-1] if len(dt) > 1 else np.array([0])
        dvy = np.diff(vy) / dt[:-1] if len(dt) > 1 else np.array([0])
        
        # Pad
        dvx = np.concatenate(([dvx[0]], dvx))
        dvy = np.concatenate(([dvy[0]], dvy))
        
        acceleration = np.sqrt(dvx**2 + dvy**2)
        
        # Calculate third derivative (jerk)
        if len(dt) > 2:
            jerk_x = np.diff(dvx) / dt[:-2] if len(dt) > 2 else np.array([0])
            jerk_y = np.diff(dvy) / dt[:-2] if len(dt) > 2 else np.array([0])
            
            jerk_x = np.concatenate(([jerk_x[0], jerk_x[0]], jerk_x))
            jerk_y = np.concatenate(([jerk_y[0], jerk_y[0]], jerk_y))
            jerk = np.sqrt(jerk_x**2 + jerk_y**2)
        else:
            jerk = np.zeros_like(acceleration)
        
        # Calculate curvature
        curvature = self._calculate_curvature_along_stroke(x, y)
        
        kinematics = {
            'velocity': {
                'x': vx.tolist(),
                'y': vy.tolist(),
                'speed': speed.tolist(),
                'mean_speed': float(np.mean(speed)),
                'max_speed': float(np.max(speed)),
                'speed_std': float(np.std(speed))
            },
            'acceleration': {
                'x': dvx.tolist(),
                'y': dvy.tolist(),
                'magnitude': acceleration.tolist(),
                'mean_accel': float(np.mean(acceleration)),
                'max_accel': float(np.max(acceleration))
            },
            'jerk': {
                'magnitude': jerk.tolist() if 'jerk' in locals() else [0] * len(stroke),
                'mean_jerk': float(np.mean(jerk)) if 'jerk' in locals() else 0.0,
                'max_jerk': float(np.max(jerk)) if 'jerk' in locals() else 0.0
            },
            'curvature': {
                'values': curvature.tolist(),
                'mean_curvature': float(np.mean(curvature)),
                'max_curvature': float(np.max(curvature)),
                'curvature_std': float(np.std(curvature))
            }
        }
        
        return kinematics
    
    def _calculate_curvature_along_stroke(self, x, y):
        """Calculate curvature along the stroke"""
        n = len(x)
        curvature = np.zeros(n)
        
        for i in range(1, n-1):
            # Three-point curvature estimation
            x0, y0 = x[i-1], y[i-1]
            x1, y1 = x[i], y[i]
            x2, y2 = x[i+1], y[i+1]
            
            # Calculate vectors
            v1 = np.array([x1 - x0, y1 - y0])
            v2 = np.array([x2 - x1, y2 - y1])
            
            # Calculate angle
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                curvature[i] = theta
        
        return curvature
    
    def detect_stroke_segments(self, stroke, min_segment_length=5):
        """Detect natural segments within a stroke"""
        if len(stroke) < min_segment_length * 2:
            return [stroke]
        
        # Use curvature changes to detect segment boundaries
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        
        curvature = self._calculate_curvature_along_stroke(x, y)
        
        # Find peaks in curvature (sharp turns)
        peaks, properties = signal.find_peaks(curvature, height=np.mean(curvature) + np.std(curvature))
        
        # Also consider low curvature points (straight segments)
        valleys, _ = signal.find_peaks(-curvature)
        
        # Combine and sort boundary points
        boundaries = sorted(set(peaks.tolist() + valleys.tolist()))
        
        # Filter boundaries to ensure minimum segment length
        filtered_boundaries = [0]
        for boundary in boundaries:
            if boundary - filtered_boundaries[-1] >= min_segment_length:
                filtered_boundaries.append(boundary)
        
        if filtered_boundaries[-1] != len(stroke) - 1:
            filtered_boundaries.append(len(stroke) - 1)
        
        # Create segments
        segments = []
        for i in range(len(filtered_boundaries) - 1):
            start = filtered_boundaries[i]
            end = filtered_boundaries[i + 1]
            if end - start >= min_segment_length:
                segments.append(stroke[start:end])
        
        return segments if segments else [stroke]
    
    def calculate_stroke_features(self, stroke):
        """Calculate comprehensive stroke features"""
        if len(stroke) < 3:
            return {}
        
        kinematics = self.analyze_stroke_kinematics(stroke)
        
        # Extract coordinates
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        
        # Basic geometric features
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        
        # Path length
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        # Straight line distance
        start_point = np.array([x[0], y[0]])
        end_point = np.array([x[-1], y[-1]])
        straight_distance = np.linalg.norm(end_point - start_point)
        
        # Linearity (how straight is the stroke)
        linearity = straight_distance / (path_length + 1e-10)
        
        # Bounding box features
        bbox_area = dx * dy
        aspect_ratio = dx / (dy + 1e-10)
        
        # Centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        
        # Moment features
        moments = self._calculate_moments(x, y)
        
        # Pressure features (if available)
        if 'pressure' in stroke[0]:
            pressures = np.array([p['pressure'] for p in stroke])
            pressure_features = {
                'mean_pressure': float(np.mean(pressures)),
                'std_pressure': float(np.std(pressures)),
                'max_pressure': float(np.max(pressures)),
                'min_pressure': float(np.min(pressures)),
                'pressure_range': float(np.max(pressures) - np.min(pressures))
            }
        else:
            pressure_features = {}
        
        # Temporal features (if timestamps available)
        temporal_features = {}
        if 'timestamp' in stroke[0]:
            timestamps = [p['timestamp'] for p in stroke]
            if hasattr(timestamps[0], 'timestamp'):
                timestamps = [ts.timestamp() for ts in timestamps]
            
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                average_speed = path_length / duration
                temporal_features = {
                    'duration': float(duration),
                    'average_speed': float(average_speed),
                    'point_frequency': len(stroke) / (duration + 1e-10)
                }
        
        features = {
            'geometric': {
                'path_length': float(path_length),
                'straight_distance': float(straight_distance),
                'linearity': float(linearity),
                'dx': float(dx),
                'dy': float(dy),
                'aspect_ratio': float(aspect_ratio),
                'bbox_area': float(bbox_area),
                'centroid': [float(centroid_x), float(centroid_y)]
            },
            'kinematic': kinematics,
            'moment': moments,
            'pressure': pressure_features,
            'temporal': temporal_features,
            'segment_count': len(self.detect_stroke_segments(stroke))
        }
        
        return features
    
    def _calculate_moments(self, x, y):
        """Calculate moment invariants"""
        # Translate to centroid
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Calculate raw moments
        m00 = len(x)  # Zero-order moment (mass)
        m10 = np.sum(x_centered)
        m01 = np.sum(y_centered)
        m20 = np.sum(x_centered**2)
        m02 = np.sum(y_centered**2)
        m11 = np.sum(x_centered * y_centered)
        
        # Calculate central moments
        mu20 = m20 / m00
        mu02 = m02 / m00
        mu11 = m11 / m00
        
        # Calculate scale invariant moments
        nu20 = mu20 / (m00**2)
        nu02 = mu02 / (m00**2)
        nu11 = mu11 / (m00**2)
        
        # Hu moments (invariant to translation, scale, rotation)
        hu1 = nu20 + nu02
        hu2 = (nu20 - nu02)**2 + 4 * nu11**2
        hu3 = (nu30 := 0)  # Higher moments would need calculation
        
        moments = {
            'raw_moments': {
                'm00': float(m00),
                'm10': float(m10),
                'm01': float(m01),
                'm20': float(m20),
                'm02': float(m02),
                'm11': float(m11)
            },
            'central_moments': {
                'mu20': float(mu20),
                'mu02': float(mu02),
                'mu11': float(mu11)
            },
            'hu_moments': {
                'hu1': float(hu1),
                'hu2': float(hu2),
                'hu3': float(hu3)
            }
        }
        
        return moments
    
    def compare_strokes(self, stroke1, stroke2):
        """Compare two strokes using Dynamic Time Warping"""
        if len(stroke1) < 3 or len(stroke2) < 3:
            return {'similarity': 0.0}
        
        # Extract features for comparison
        x1 = np.array([p['x'] for p in stroke1])
        y1 = np.array([p['y'] for p in stroke1])
        x2 = np.array([p['x'] for p in stroke2])
        y2 = np.array([p['y'] for p in stroke2])
        
        # Create feature vectors (x, y coordinates)
        features1 = np.column_stack((x1, y1))
        features2 = np.column_stack((x2, y2))
        
        # Calculate DTW distance
        dtw_distance = self._dtw_distance(features1, features2)
        
        # Normalize by stroke lengths
        max_length = max(len(stroke1), len(stroke2))
        normalized_distance = dtw_distance / max_length if max_length > 0 else 1.0
        
        # Convert to similarity (0-1, where 1 is identical)
        similarity = 1.0 / (1.0 + normalized_distance)
        
        return {
            'dtw_distance': float(dtw_distance),
            'normalized_distance': float(normalized_distance),
            'similarity': float(similarity),
            'length_ratio': len(stroke1) / (len(stroke2) + 1e-10)
        }
    
    def _dtw_distance(self, series1, series2):
        """Dynamic Time Warping distance between two series"""
        n, m = len(series1), len(series2)
        dtw_matrix = np.zeros((n+1, m+1))
        
        # Initialize with infinity
        dtw_matrix.fill(np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill the matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(series1[i-1] - series2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[n, m]
    
    def visualize_stroke(self, stroke, ax=None):
        """Visualize stroke with kinematic information"""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot stroke path
        x = [p['x'] for p in stroke]
        y = [p['y'] for p in stroke]
        
        # Create color gradient based on speed
        kinematics = self.analyze_stroke_kinematics(stroke)
        if 'velocity' in kinematics:
            speeds = kinematics['velocity']['speed']
            
            # Normalize speeds for coloring
            if len(speeds) > 0:
                norm_speeds = (speeds - np.min(speeds)) / (np.max(speeds) - np.min(speeds) + 1e-10)
                
                # Create line segments with colors
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
                lc.set_array(norm_speeds)
                lc.set_linewidth(2)
                ax[0].add_collection(lc)
                
                # Add colorbar
                plt.colorbar(lc, ax=ax[0], label='Normalized Speed')
            else:
                ax[0].plot(x, y, 'b-', linewidth=2)
        else:
            ax[0].plot(x, y, 'b-', linewidth=2)
        
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_title('Stroke Path with Speed Coloring')
        ax[0].axis('equal')
        ax[0].grid(True, alpha=0.3)
        
        # Plot kinematic profiles
        if 'velocity' in kinematics and 'acceleration' in kinematics:
            t = np.arange(len(stroke))
            
            ax[1].plot(t, kinematics['velocity']['speed'], 'g-', label='Speed', linewidth=2)
            ax[1].plot(t, kinematics['acceleration']['magnitude'], 'r-', label='Acceleration', linewidth=2)
            
            if 'jerk' in kinematics:
                ax[1].plot(t, kinematics['jerk']['magnitude'], 'b-', label='Jerk', linewidth=2)
            
            ax[1].set_xlabel('Point Index')
            ax[1].set_ylabel('Magnitude')
            ax[1].set_title('Kinematic Profiles')
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return ax