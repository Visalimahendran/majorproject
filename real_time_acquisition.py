import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import json
import os

class RealTimeHandwritingCapture:
    """Capture real-time handwriting from webcam"""
    
    def __init__(self, camera_id=0, resolution=(1280, 720)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.is_capturing = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.analysis_thread = None
        
        # Writing detection parameters
        self.writing_detected = False
        self.last_writing_time = 0
        self.writing_frames = []
        self.max_frames = 100
        
        # ROI (Region of Interest) for writing area
        self.roi = None
        self.roi_set = False
        
    def start_capture(self):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_id}")
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.start()
        
        print(f"✅ Started capture from camera {self.camera_id}")
        return True
    
    def _capture_frames(self):
        """Continuously capture frames"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    timestamp = time.time()
                    self.frame_queue.put((frame, timestamp))
                
                # Detect writing in frame
                if self.roi_set:
                    writing_frame = self.detect_writing(frame)
                    if writing_frame is not None:
                        self.writing_detected = True
                        self.last_writing_time = time.time()
                        self.writing_frames.append(writing_frame)
                        
                        # Keep only recent frames
                        if len(self.writing_frames) > self.max_frames:
                            self.writing_frames = self.writing_frames[-self.max_frames:]
            else:
                break
            
            time.sleep(0.033)  # ~30 FPS
    
    def detect_writing(self, frame):
        """Detect handwriting in the frame"""
        if self.roi is None:
            return None
        
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for writing-like contours
        writing_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Reasonable size for writing
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.5:  # Not too circular (likely writing)
                        writing_contours.append(contour)
        
        if writing_contours:
            # Create mask for writing
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, writing_contours, -1, 255, -1)
            
            # Apply mask to original ROI
            result = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
            return result
        
        return None
    
    def set_roi(self, x, y, w, h):
        """Set Region of Interest for writing detection"""
        self.roi = (x, y, w, h)
        self.roi_set = True
        print(f"✅ ROI set to: {self.roi}")
    
    def get_frame(self, timeout=1):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def get_writing_frames(self):
        """Get captured writing frames"""
        frames = self.writing_frames.copy()
        self.writing_frames.clear()
        return frames
    
    def is_writing_active(self, threshold=2):
        """Check if writing is currently active"""
        return (time.time() - self.last_writing_time) < threshold
    
    def stop_capture(self):
        """Stop webcam capture"""
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Capture stopped")


class DigitalWritingCapture:
    """Capture digital writing from input devices"""
    
    def __init__(self):
        self.strokes = []
        self.current_stroke = []
        self.is_recording = False
        self.start_time = None
        self.sampling_rate = 100  # Hz
        
    def start_recording(self):
        """Start recording digital writing"""
        self.is_recording = True
        self.start_time = time.time()
        self.strokes = []
        print("✅ Started digital writing recording")
    
    def add_point(self, x, y, pressure=0.5):
        """Add a point to current stroke"""
        if not self.is_recording:
            return False
        
        timestamp = time.time() - self.start_time
        point = {
            'x': x,
            'y': y,
            'pressure': pressure,
            'timestamp': timestamp,
            'time_delta': 1.0 / self.sampling_rate
        }
        
        self.current_stroke.append(point)
        return True
    
    def end_stroke(self):
        """End current stroke and start new one"""
        if self.current_stroke:
            self.strokes.append(self.current_stroke.copy())
            self.current_stroke = []
            return True
        return False
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
        
        print(f"✅ Stopped recording. Captured {len(self.strokes)} strokes")
        return self.get_recording_data()
    
    def get_recording_data(self):
        """Get recording data in structured format"""
        data = {
            'recording_id': f"digital_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': self.start_time,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'total_points': sum(len(stroke) for stroke in self.strokes),
            'stroke_count': len(self.strokes),
            'strokes': self.strokes,
            'sampling_rate': self.sampling_rate
        }
        return data
    
    def save_recording(self, filepath):
        """Save recording to file"""
        data = self.get_recording_data()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved recording to {filepath}")
    
    def load_recording(self, filepath):
        """Load recording from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.strokes = data.get('strokes', [])
        print(f"✅ Loaded recording with {len(self.strokes)} strokes")
        return data
    
    def simulate_writing(self, text="Sample Writing"):
        """Simulate digital writing for testing"""
        self.start_recording()
        
        # Simulate writing each character
        for i, char in enumerate(text):
            if char == ' ':
                # Space - end stroke and pause
                self.end_stroke()
                time.sleep(0.1)
                continue
            
            # Start new stroke for character
            base_x = 100 + i * 30
            base_y = 200
            
            # Simulate character strokes
            if char in 'iljft':
                # Vertical strokes
                for y in range(0, 40, 2):
                    self.add_point(base_x + 10, base_y + y, 0.5 + np.random.random() * 0.2)
            else:
                # More complex strokes
                for x in range(0, 20, 2):
                    y = np.sin(x * 0.2) * 10
                    self.add_point(base_x + x, base_y + y, 0.5 + np.random.random() * 0.2)
            
            self.end_stroke()
            time.sleep(0.05)
        
        return self.stop_recording()