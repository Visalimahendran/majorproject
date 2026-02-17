import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import threading
import queue
import time

class WebcamCapture:
    """Capture and process webcam data for paper handwriting"""
    
    def __init__(self, device_id=0, width=1920, height=1080, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize video capture
        self.cap = None
        self.is_capturing = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Calibration data
        self.calibration_points = []
        self.homography_matrix = None
        
    def start_capture(self):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open webcam {self.device_id}")
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.start()
        
        return True
    
    def _capture_frames(self):
        """Continuously capture frames"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                break
            time.sleep(1/self.fps)
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None
    
    def detect_handwriting(self, frame):
        """Detect handwriting from webcam frame"""
        if frame is None:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        handwriting_data = {
            'timestamp': datetime.now(),
            'frame': frame,
            'hand_landmarks': None,
            'writing_region': None,
            'pen_tip': None
        }
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                    landmarks.append((x, y, idx))
                
                handwriting_data['hand_landmarks'] = landmarks
                
                # Detect pen position (index finger tip)
                index_tip = landmarks[8]  # Index finger tip
                handwriting_data['pen_tip'] = (index_tip[0], index_tip[1])
                
                # Detect writing region
                writing_region = self._detect_writing_region(landmarks)
                handwriting_data['writing_region'] = writing_region
        
        return handwriting_data
    
    def _detect_writing_region(self, landmarks):
        """Detect the writing region based on hand position"""
        # Get bounding box of hand
        xs = [l[0] for l in landmarks]
        ys = [l[1] for l in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Expand region slightly
        padding = 50
        return (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(self.width, x_max + padding),
            min(self.height, y_max + padding)
        )
    
    def calibrate_paper(self, calibration_image):
        """Calibrate for paper-based writing"""
        # Detect paper corners
        gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (paper)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                # Found paper rectangle
                src_points = np.float32([point[0] for point in approx])
                
                # Destination points (A4 paper size at 300 DPI)
                dst_points = np.float32([
                    [0, 0],
                    [2480, 0],  # A4 width at 300 DPI
                    [2480, 3508],  # A4 height at 300 DPI
                    [0, 3508]
                ])
                
                # Calculate homography matrix
                self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                return True
        
        return False
    
    def transform_to_paper_coords(self, point):
        """Transform screen coordinates to paper coordinates"""
        if self.homography_matrix is None:
            return point
        
        src_point = np.array([[point[0], point[1]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point.reshape(1, -1, 2), self.homography_matrix)
        
        return (int(dst_point[0][0][0]), int(dst_point[0][0][1]))
    
    def extract_strokes(self, frame_sequence):
        """Extract handwriting strokes from frame sequence"""
        strokes = []
        current_stroke = []
        pen_down = False
        
        for frame_data in frame_sequence:
            if frame_data['pen_tip']:
                point = self.transform_to_paper_coords(frame_data['pen_tip'])
                timestamp = frame_data['timestamp']
                
                # Detect pen down/up based on velocity
                if not current_stroke:
                    # Start new stroke
                    current_stroke.append({
                        'x': point[0],
                        'y': point[1],
                        'timestamp': timestamp,
                        'pressure': 0.5  # Estimated
                    })
                    pen_down = True
                else:
                    # Check if pen lifted
                    last_point = current_stroke[-1]
                    distance = np.sqrt((point[0] - last_point['x'])**2 + 
                                     (point[1] - last_point['y'])**2)
                    
                    if distance < 5:  # Too small movement
                        if len(current_stroke) > 1:
                            strokes.append(current_stroke)
                        current_stroke = []
                        pen_down = False
                    else:
                        # Continue stroke
                        current_stroke.append({
                            'x': point[0],
                            'y': point[1],
                            'timestamp': timestamp,
                            'pressure': 0.5
                        })
        
        # Add last stroke if exists
        if current_stroke and len(current_stroke) > 1:
            strokes.append(current_stroke)
        
        return strokes
    
    def stop_capture(self):
        """Stop webcam capture"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        
        cv2.destroyAllWindows()
    
    def get_calibration_frame(self):
        """Get frame for calibration"""
        frame = self.get_frame()
        if frame is not None:
            # Add calibration markers
            h, w = frame.shape[:2]
            markers = [
                (w//4, h//4),
                (3*w//4, h//4),
                (3*w//4, 3*h//4),
                (w//4, 3*h//4)
            ]
            
            for marker in markers:
                cv2.circle(frame, marker, 10, (0, 255, 0), 2)
                cv2.putText(frame, "Calibration Point", 
                          (marker[0] - 100, marker[1] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame