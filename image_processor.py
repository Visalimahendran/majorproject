import cv2
import numpy as np
from PIL import Image
import skimage
from skimage import filters, morphology, exposure
import warnings
warnings.filterwarnings('ignore')

class ImageProcessor:
    """Process images for handwriting analysis"""
    
    def __init__(self, target_size=(224, 224), grayscale=True, normalize=True):
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        
    def process(self, image):
        """Process image through pipeline"""
        if isinstance(image, str):
            # Load from file
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply processing pipeline
        processed = self._pipeline(image)
        
        return processed
    
    def _pipeline(self, image):
        """Image processing pipeline"""
        # 1. Convert to grayscale if requested
        if self.grayscale:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize
        image = cv2.resize(image, self.target_size)
        
        # 3. Noise removal
        image = self._remove_noise(image)
        
        # 4. Binarization (if grayscale)
        if self.grayscale:
            image = self._binarize(image)
        
        # 5. Normalize
        if self.normalize:
            image = self._normalize_image(image)
        
        # 6. Skeletonization (for stroke analysis)
        if self.grayscale:
            image = self._skeletonize(image)
        
        return image
    
    def _remove_noise(self, image):
        """Remove noise from image"""
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        # Additional Gaussian blur for smoothness
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        return denoised
    
    def _binarize(self, image):
        """Binarize grayscale image"""
        # Adaptive thresholding for varying lighting
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up small noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _normalize_image(self, image):
        """Normalize image intensities"""
        if len(image.shape) == 3:
            # Color image - normalize each channel
            normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                channel = image[:, :, i].astype(np.float32)
                if np.std(channel) > 0:
                    channel = (channel - np.mean(channel)) / np.std(channel)
                normalized[:, :, i] = channel
        else:
            # Grayscale image
            normalized = image.astype(np.float32)
            if np.std(normalized) > 0:
                normalized = (normalized - np.mean(normalized)) / np.std(normalized)
        
        # Scale to 0-1 range
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        
        # Convert back to uint8 for compatibility
        if image.dtype == np.uint8:
            normalized = (normalized * 255).astype(np.uint8)
        
        return normalized
    
    def _skeletonize(self, image):
        """Skeletonize binary image for stroke analysis"""
        if image.max() > 1:
            # Binarize if not already binary
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = (image * 255).astype(np.uint8)
        
        # Skeletonization using morphological operations
        skeleton = np.zeros_like(binary)
        
        # Zhang-Suen thinning algorithm
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        while not done:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            done = cv2.countNonZero(binary) == 0
        
        return skeleton
    
    def extract_text_regions(self, image):
        """Extract text regions from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Enhanced edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect text components
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio (text-like regions)
        text_regions = []
        min_area = 100
        max_area = image.shape[0] * image.shape[1] * 0.8
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Text typically has aspect ratio > 1 (wider than tall)
                if aspect_ratio > 0.2 and aspect_ratio < 10:
                    text_regions.append((x, y, w, h))
        
        return text_regions
    
    def segment_characters(self, image):
        """Segment individual characters"""
        # Binarize image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        characters = []
        min_area = 50
        max_area = image.shape[0] * image.shape[1] * 0.1
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Extract character region
                char_region = image[y:y+h, x:x+w]
                
                # Resize to standard size
                char_region = cv2.resize(char_region, (32, 32))
                
                characters.append({
                    'region': char_region,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': centroids[i]
                })
        
        # Sort characters left to right
        characters.sort(key=lambda c: c['bbox'][0])
        
        return characters
    
    def extract_stroke_features(self, skeleton):
        """Extract features from skeletonized image"""
        if len(skeleton.shape) == 3:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
        
        # Ensure binary
        _, binary = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours (strokes)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'stroke_count': len(contours),
            'total_length': 0,
            'avg_stroke_width': 0,
            'stroke_density': 0,
            'curvature_features': []
        }
        
        stroke_lengths = []
        stroke_widths = []
        
        for contour in contours:
            # Stroke length
            length = cv2.arcLength(contour, False)
            stroke_lengths.append(length)
            features['total_length'] += length
            
            # Stroke width estimation
            area = cv2.contourArea(contour)
            if length > 0:
                width = area / length
                stroke_widths.append(width)
            
            # Curvature features
            if len(contour) > 5:
                curvature = self._calculate_curvature(contour)
                features['curvature_features'].append(curvature)
        
        if stroke_lengths:
            features['avg_stroke_length'] = np.mean(stroke_lengths)
            features['std_stroke_length'] = np.std(stroke_lengths)
        
        if stroke_widths:
            features['avg_stroke_width'] = np.mean(stroke_widths)
            features['std_stroke_width'] = np.std(stroke_widths)
        
        # Stroke density (pixels per area)
        features['stroke_density'] = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
        
        return features
    
    def _calculate_curvature(self, contour):
        """Calculate curvature of a contour"""
        # Resample contour for consistent spacing
        contour = contour.squeeze()
        if len(contour) < 10:
            return 0
        
        # Calculate derivatives
        dx = np.gradient(contour[:, 0])
        dy = np.gradient(contour[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5)
        
        # Avoid division by zero
        denominator[denominator == 0] = 1e-10
        curvature = numerator / denominator
        
        return {
            'mean_curvature': np.mean(curvature),
            'std_curvature': np.std(curvature),
            'max_curvature': np.max(curvature)
        }
    
    def save_processed(self, image, filename):
        """Save processed image"""
        if len(image.shape) == 2:
            # Grayscale
            cv2.imwrite(filename, image)
        else:
            # Color
            if image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, image_bgr)
            else:
                cv2.imwrite(filename, image)
        
        print(f"Saved processed image to {filename}")