import cv2
import numpy as np
from datetime import datetime
import time
import queue
import threading
from collections import deque

class VideoProcessor:
    """Process video streams for real-time handwriting analysis"""
    
    def __init__(self, frame_skip=2, roi_size=(500, 500), min_contour_area=100):
        self.frame_skip = frame_skip
        self.roi_size = roi_size
        self.min_contour_area = min_contour_area
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Optical flow
        self.prev_gray = None
        self.flow_points = deque(maxlen=1000)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=30)  # Keep last 30 frames
        self.processing_queue = queue.Queue()
        
    def process_frame(self, frame):
        """Process a single video frame"""
        if frame is None:
            return None
        
        # Store in buffer
        self.frame_buffer.append({
            'frame': frame.copy(),
            'timestamp': datetime.now()
        })
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (640, 480))
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Process results
        results = {
            'original': frame,
            'gray': gray,
            'blurred': blurred,
            'timestamp': datetime.now(),
            'motion_detected': False,
            'writing_region': None,
            'optical_flow': None
        }
        
        # Detect motion
        motion_mask = self._detect_motion(blurred)
        if motion_mask is not None:
            results['motion_mask'] = motion_mask
            results['motion_detected'] = np.sum(motion_mask > 0) > 100
        
        # Detect writing hand
        writing_region = self._detect_writing_hand(frame)
        if writing_region:
            results['writing_region'] = writing_region
        
        # Calculate optical flow
        flow = self._calculate_optical_flow(gray)
        if flow is not None:
            results['optical_flow'] = flow
        
        # Extract potential strokes
        if results['motion_detected'] and writing_region:
            strokes = self._extract_strokes_from_motion(motion_mask, writing_region)
            results['potential_strokes'] = strokes
        
        return results
    
    def _detect_motion(self, frame):
        """Detect motion in frame using background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Threshold to get binary mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        large_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                large_contours.append(contour)
        
        # Create mask with only large contours
        motion_mask = np.zeros_like(thresh)
        cv2.drawContours(motion_mask, large_contours, -1, 255, -1)
        
        return motion_mask if np.sum(motion_mask) > 0 else None
    
    def _detect_writing_hand(self, frame):
        """Detect writing hand using color and shape features"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range (adjust based on lighting)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in skin mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (likely hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand region slightly
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Check if region looks like a hand (aspect ratio)
        aspect_ratio = w / h
        if 0.5 < aspect_ratio < 2.0:
            return (x, y, w, h)
        
        return None
    
    def _calculate_optical_flow(self, gray_frame):
        """Calculate optical flow for motion analysis"""
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return None
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_frame,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Update previous frame
        self.prev_gray = gray_frame.copy()
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize for visualization
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create HSV image for flow visualization
        hsv = np.zeros((gray_frame.shape[0], gray_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
        hsv[..., 1] = 255  # Saturation
        hsv[..., 2] = magnitude_norm.astype(np.uint8)  # Value
        
        # Convert to BGR for display
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Track flow points for stroke detection
        self._track_flow_points(flow, magnitude)
        
        return {
            'flow': flow,
            'magnitude': magnitude,
            'angle': angle,
            'visualization': flow_bgr,
            'mean_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude)
        }
    
    def _track_flow_points(self, flow, magnitude):
        """Track points with significant motion"""
        # Create grid of points
        h, w = flow.shape[:2]
        step = 10
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                mag = magnitude[y, x]
                if mag > 1.0:  # Significant motion threshold
                    dx, dy = flow[y, x]
                    self.flow_points.append({
                        'x': x,
                        'y': y,
                        'dx': dx,
                        'dy': dy,
                        'magnitude': mag,
                        'timestamp': time.time()
                    })
    
    def _extract_strokes_from_motion(self, motion_mask, writing_region):
        """Extract potential strokes from motion in writing region"""
        x, y, w, h = writing_region
        
        # Extract region of interest
        roi_mask = motion_mask[y:y+h, x:x+w]
        
        if np.sum(roi_mask) == 0:
            return []
        
        # Find contours in ROI
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        strokes = []
        min_stroke_length = 20
        
        for contour in contours:
            # Filter by size
            if cv2.contourArea(contour) < 50:
                continue
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:  # At least a triangle
                # Get bounding box in original coordinates
                contour_global = approx + np.array([x, y])
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour_global)
                
                # Check if contour resembles a stroke
                aspect_ratio = w_c / h_c
                if 0.2 < aspect_ratio < 5.0:  # Reasonable stroke aspect ratio
                    # Calculate stroke features
                    length = cv2.arcLength(contour, True)
                    if length > min_stroke_length:
                        strokes.append({
                            'contour': contour_global,
                            'bbox': (x_c, y_c, w_c, h_c),
                            'length': length,
                            'area': cv2.contourArea(contour),
                            'center': (x_c + w_c//2, y_c + h_c//2)
                        })
        
        return strokes
    
    def process_video_file(self, video_path, callback=None):
        """Process video file frame by frame"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        processed_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_count % self.frame_skip == 0:
                results = self.process_frame(frame)
                processed_frames.append(results)
                
                if callback:
                    callback(results, frame_count)
            
            frame_count += 1
        
        cap.release()
        return processed_frames
    
    def realtime_processing(self, camera_id=0, processing_callback=None):
        """Real-time video processing from webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        print(f"Starting real-time processing from camera {camera_id}")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            frame_count += 1
            
            # Display results
            display_frame = self._create_display_frame(frame, results)
            cv2.imshow('NeuroMotor Video Processing', display_frame)
            
            # Call processing callback
            if processing_callback:
                processing_callback(results)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _create_display_frame(self, original_frame, results):
        """Create display frame with annotations"""
        display = original_frame.copy()
        
        # Draw writing region
        if results.get('writing_region'):
            x, y, w, h = results['writing_region']
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Writing Region", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw motion detection
        if results.get('motion_detected'):
            cv2.putText(display, "Motion Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw strokes
        if 'potential_strokes' in results:
            for stroke in results['potential_strokes']:
                cv2.drawContours(display, [stroke['contour']], -1, (255, 0, 0), 2)
                center = stroke['center']
                cv2.putText(display, f"L:{stroke['length']:.0f}", 
                           (center[0], center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Display frame info
        cv2.putText(display, f"Frame: {len(self.frame_buffer)}", 
                   (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def get_stroke_trajectories(self, min_frames=5):
        """Extract stroke trajectories from flow points"""
        if len(self.flow_points) < min_frames:
            return []
        
        # Group points into trajectories based on proximity
        trajectories = []
        used_points = set()
        
        for i, point in enumerate(self.flow_points):
            if i in used_points:
                continue
            
            trajectory = [point]
            used_points.add(i)
            
            # Find connected points
            current_point = point
            for j, other_point in enumerate(self.flow_points):
                if j in used_points:
                    continue
                
                # Calculate distance
                dist = np.sqrt((other_point['x'] - current_point['x'])**2 + 
                              (other_point['y'] - current_point['y'])**2)
                
                # Check time difference
                time_diff = abs(other_point['timestamp'] - current_point['timestamp'])
                
                if dist < 20 and time_diff < 0.5:  # Connected points
                    trajectory.append(other_point)
                    used_points.add(j)
                    current_point = other_point
            
            if len(trajectory) >= min_frames:
                trajectories.append(trajectory)
        
        return trajectories