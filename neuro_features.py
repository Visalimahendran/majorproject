import numpy as np
from scipy import signal, stats

class NeuroFeatureExtractor:
    """Extract neuro-motor features from handwriting"""
    
    def __init__(self, tremor_range=(3, 12)):
        self.tremor_range = tremor_range
        
    def extract(self, strokes, temporal_features):
        """Extract all neuro-motor features"""
        features = {}
        
        # Extract features from each stroke
        all_tremors = []
        all_jitters = []
        all_irregularities = []
        
        for stroke in strokes:
            if len(stroke) < 10:
                continue
                
            # Get coordinates
            x = [p['x'] for p in stroke]
            y = [p['y'] for p in stroke]
            t = [p['time_elapsed'] for p in stroke]
            
            # Calculate features
            tremor = self.calculate_tremor(x, y, t)
            jitter = self.calculate_jitter(x, y)
            irregularity = self.calculate_irregularity(x, y)
            
            if tremor is not None:
                all_tremors.append(tremor)
            if jitter is not None:
                all_jitters.append(jitter)
            if irregularity is not None:
                all_irregularities.append(irregularity)
        
        # Aggregate features
        features['tremor_intensity'] = np.mean(all_tremors) if all_tremors else 0
        features['jitter_index'] = np.mean(all_jitters) if all_jitters else 0
        features['slant_variability'] = np.std(all_irregularities) if all_irregularities else 0
        features['micro_movements'] = self.calculate_micro_movements(strokes)
        features['hesitation_patterns'] = self.detect_hesitations(temporal_features)
        
        return features
    
    def calculate_tremor(self, x, y, t):
        """Calculate tremor intensity using frequency analysis"""
        if len(x) < 20:
            return None
            
        # Calculate velocity
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Avoid division by zero
        dt[dt == 0] = 0.001
        
        vx = dx / dt
        vy = dy / dt
        velocity = np.sqrt(vx**2 + vy**2)
        
        # Remove NaN values
        velocity = velocity[~np.isnan(velocity)]
        if len(velocity) < 10:
            return None
        
        # Frequency analysis
        try:
            fs = 1.0 / np.mean(dt[dt > 0])  # Sampling frequency
            f, Pxx = signal.periodogram(velocity, fs)
            
            # Filter for tremor range (3-12 Hz)
            tremor_mask = (f >= self.tremor_range[0]) & (f <= self.tremor_range[1])
            if np.any(tremor_mask):
                tremor_power = np.mean(Pxx[tremor_mask])
                return tremor_power * 1000  # Scale for readability
        except:
            return None
        
        return 0
    
    def calculate_jitter(self, x, y):
        """Calculate movement jitter"""
        if len(x) < 5:
            return None
            
        # Calculate smoothness of movement
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Second derivative (acceleration)
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        if len(ddx) < 2:
            return None
            
        jitter = np.sqrt(np.mean(ddx**2 + ddy**2))
        return jitter * 100  # Scale for readability
    
    def calculate_irregularity(self, x, y):
        """Calculate stroke irregularity"""
        if len(x) < 10:
            return None
            
        # Fit a smooth curve and calculate residuals
        t = np.arange(len(x))
        
        try:
            # Linear fit
            coeff_x = np.polyfit(t, x, 1)
            coeff_y = np.polyfit(t, y, 1)
            
            # Predict smooth curve
            x_smooth = np.polyval(coeff_x, t)
            y_smooth = np.polyval(coeff_y, t)
            
            # Calculate irregularity as RMS error
            irregularity = np.sqrt(np.mean((x - x_smooth)**2 + (y - y_smooth)**2))
            return irregularity
        except:
            return None
    
    def calculate_micro_movements(self, strokes):
        """Count micro-movements (small direction changes)"""
        micro_movements = 0
        
        for stroke in strokes:
            if len(stroke) < 5:
                continue
                
            for i in range(2, len(stroke)):
                # Calculate angle change
                dx1 = stroke[i-1]['x'] - stroke[i-2]['x']
                dy1 = stroke[i-1]['y'] - stroke[i-2]['y']
                dx2 = stroke[i]['x'] - stroke[i-1]['x']
                dy2 = stroke[i]['y'] - stroke[i-1]['y']
                
                # Avoid division by zero
                if dx1 == 0 and dy1 == 0:
                    continue
                if dx2 == 0 and dy2 == 0:
                    continue
                
                # Calculate angle change (in degrees)
                angle1 = np.degrees(np.arctan2(dy1, dx1))
                angle2 = np.degrees(np.arctan2(dy2, dx2))
                angle_change = abs(angle2 - angle1) % 360
                angle_change = min(angle_change, 360 - angle_change)
                
                # Count as micro-movement if small angle change
                if 5 < angle_change < 30:
                    micro_movements += 1
        
        return micro_movements
    
    def detect_hesitations(self, temporal_features):
        """Detect hesitation patterns from temporal features"""
        hesitations = 0
        
        if 'pauses' in temporal_features:
            pauses = temporal_features['pauses']
            # Count significant pauses (> 0.3 seconds)
            hesitations = sum(1 for pause in pauses if pause > 0.3)
        
        return hesitations