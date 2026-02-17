import numpy as np
from datetime import datetime, time

class DigitalCanvas:
    """Handle digital handwriting input"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.strokes = []
        self.current_stroke = []
        self.start_time = None
        
    def start_stroke(self, x, y, pressure=0.5):
        """Start a new stroke"""
        self.current_stroke = []
        self.start_time = datetime.now()
        self.add_point(x, y, pressure)
        
    def add_point(self, x, y, pressure=0.5):
        """Add point to current stroke"""
        timestamp = datetime.now()
        point = {
            'x': x,
            'y': y,
            'pressure': pressure,
            'timestamp': timestamp,
            'time_elapsed': (timestamp - self.start_time).total_seconds() if self.start_time else 0
        }
        self.current_stroke.append(point)
        return point
    
    def end_stroke(self):
        """End current stroke"""
        if self.current_stroke:
            self.strokes.append(self.current_stroke.copy())
            self.current_stroke = []
            return True
        return False
    
    def get_strokes(self):
        """Get all strokes"""
        return self.strokes.copy()
    
    def clear(self):
        """Clear all strokes"""
        self.strokes = []
        self.current_stroke = []
        
    def capture_sentence(self, sentence):
        """Simulate sentence writing for testing"""
        words = sentence.split()
        for i, word in enumerate(words):
            self.start_stroke(50 + i*100, 300)
            # Simulate writing word
            for j, char in enumerate(word):
                self.add_point(50 + i*100 + j*20, 300 + np.random.normal(0, 5))
            self.end_stroke()
            # Simulate space
            if i < len(words) - 1:
                time.sleep(0.1)
        
        return self.get_strokes()