import numpy as np
import cv2

class PlantDetection:
    def __init__(self, fx, fy, cx, cy, x_min, x_max, y_min, y_max, z_min, z_max, detection_threshold=0.15, min_points_threshold=500):
        """
        Initialize the plant detector class
        """
        # Camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Bounding box params
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        
        # Detection params
        self.detection_threshold = detection_threshold
        self.min_points_threshold = min_points_threshold
        
        # Detection state
        self.is_detected = False
        self.detection_confidence = 0.0
        self.points_in_box = 0
        
    def detect_plant_vectorized(self, depth_map):
        """
        Highly optimized vectorized detection for maximum performance in
        live streaming.
        
        This is the fastest method and recommended for real-time application.
        """
        h, w = depth_map.shape
        
        # Create coords grids
        v_grid, u_grid = np.mgrid[0:h, 0:w]
        
        # Get valid depth mask
        valid_mask = depth_map > 0
        
        # convert to meters
        z = depth_map.astype(np.float32) / 1000.0
        
        # Vectorized conversion to 3D coords
        x = (u_grid - self.cx) * z / self.fx
        y = (v_grid - self.cy) * z / self.fy
        
        # Vectorized bounding box check
        in_bounds = (valid_mask & (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max) & (z >= self.z_min) & (z <= self.z_max))
        
        # Count points in bounds
        points_in_bound = np.sum(in_bounds)
        
        # Calculate detection confidence
        bbox_volume = (self.x_max - self.x_min) * \
                      (self.y_max - self.y_min) * \
                      (self.z_max - self.z_min)
                      
        avg_point_spacing = 0.005
        expected_points = bbox_volume / (avg_point_spacing ** 3)
        confidence = min(1.0, float(points_in_bounds) / expected_points)
        
        # Detection decision
        detected = (points_in_bounds >= self.min_points_threshold) and \
                   (confidence >= self.detection_threshold)
                   
        # Update state
        self.is_detected = detected
        self.detection_confidence = confidence
        self.points_in_box = int(points_in_bounds)
        
        return detected, confidence, int(points_in_bounds)
        
    def detect_plant(depth_map):
        """
        Detect if a plant/object isin the center frame
        Use the Vectorized methods
        """
        return self.detect_plant_vectorized(depth_map)
        
        
