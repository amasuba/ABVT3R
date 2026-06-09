import numpy as np
from sklearn.neighbors import NearestNeighbors

class PreProcessing:
    """
    A comprehensive point cloud preprocessing class for 3D computer vision applications.
    
    This class provides methods for converting depth maps to point clouds and applying
    various filtering and smoothing techniques including:
    - Point cloud generation from depth maps
    - PassThrough filtering for region of interest selection
    - Statistical outlier removal
    - Moving Least Squares (MLS) surface smoothing
    """
    
    def __init__(self):
        """Initialize the PreProcessing class."""
        pass
    
    def point_cloud_generation(self, depth_map, fx, fy, cx, cy):
        # Step 1: Convert depth map units from mm to meters
        depth_meters = depth_map * 0.001
        
        # Step 2: Generate (u,v) coordinates for each pixel
        height, width = depth_map.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Step 3: Flatten coordinate arrays for vectorized processing
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_meters.flatten()
        
        # Step 4: Filter based on reasonable depth range
        min_depth = 0.1  # 10cm minimum distance
        max_depth = 2.5  # 2.5m maximum distance
        
        valid_mask = (depth_flat > min_depth) & (depth_flat < max_depth) & (depth_flat > 0)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        
        # Step 5: Normalize with camera intrinsic parameters
        x_normalized = (u_valid - cx) / fx
        y_normalized = (v_valid - cy) / fy
        
        # Step 6: Generate 3D world coordinates
        x = x_normalized * depth_valid
        y = -y_normalized * depth_valid  # Flip Y axis for standard coordinate system
        z = depth_valid  # Negative Z for forward-facing camera
        
        # Step 7: Combine to create point cloud array
        array_3d = np.column_stack((x, y, z))
        valid_pixels = np.column_stack((u_valid, v_valid))
        
        return array_3d, valid_pixels
    
    def pass_through_filter(self, array_3d, valid_pixels, x_min, x_max, y_min, y_max, z_min, z_max):
        # Create filter mask for all three dimensions
        filter_mask = ((array_3d[:, 0] > x_min) & (array_3d[:, 0] < x_max) & 
                      (array_3d[:, 1] > y_min) & (array_3d[:, 1] < y_max) & 
                      (array_3d[:, 2] > z_min) & (array_3d[:, 2] < z_max))
        
        # Apply mask to filter points and pixels
        filtered_points = array_3d[filter_mask]
        filtered_pixels = valid_pixels[filter_mask]
        
        return filtered_points, filtered_pixels
    
    def statistical_outlier_removal(self, filtered_points, filtered_pixels, m=50, k=1.0):
        # Step 1: Build nearest neighbor tree
        nn_tree = NearestNeighbors(n_neighbors=m + 1, algorithm='kd_tree')
        nn_tree.fit(filtered_points)
        
        # Step 2: Calculate neighbor distances
        distances, indices = nn_tree.kneighbors(filtered_points)
        
        # Remove self from calculation and compute average distances
        neighbor_dist = distances[:, 1:]  # Exclude self (first neighbor)
        avg_dist = np.mean(neighbor_dist, axis=1)
        
        # Step 3: Statistical analysis and threshold determination
        mu = avg_dist.mean()
        sigma = avg_dist.std()
        
        # Define acceptable range based on Gaussian distribution
        dist_min = mu - k * sigma
        dist_max = mu + k * sigma
        
        # Step 4: Apply statistical filtering
        stat_mask = (avg_dist >= dist_min) & (avg_dist <= dist_max)
        
        cleaned_points = filtered_points[stat_mask]
        cleaned_pixels = filtered_pixels[stat_mask]  # Fixed: was filtered_points[stat_mask]
        
        return cleaned_points, cleaned_pixels
    
    def surface_smoothing_mls(self, cleaned_points, cleaned_pixels, search_radius=0.0005, order=2):
        # Step 1: Build KD-Tree for neighborhood search
        nn_tree = NearestNeighbors(radius=search_radius, algorithm='kd_tree')
        nn_tree.fit(cleaned_points)
        
        # Step 2: Process points in batches for memory efficiency
        batch_size = 1000
        n_points = len(cleaned_points)
        n_batches = (n_points + batch_size - 1) // batch_size
        
        smoothed_points = []
        surface_normals = []
        
        # Determine required coefficients based on polynomial order
        n_coeffs = 6 if order == 2 else 3
        
        # Step 3: Batch processing for scalability
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_points)
            batch_points = cleaned_points[start_idx:end_idx]
            
            # Find neighbors for current batch
            batch_neighbor_indices = nn_tree.radius_neighbors(batch_points, return_distance=False)
            
            # Step 4: Fit polynomial surfaces for each point
            for point, neighbor_indices in zip(batch_points, batch_neighbor_indices):
                if len(neighbor_indices) < n_coeffs:
                    # Insufficient neighbors - keep original point
                    smoothed_points.append(point)
                    surface_normals.append([0, 0, 1])  # Default normal
                    continue
                
                # Extract neighborhood coordinates
                neighbors = cleaned_points[neighbor_indices]
                x_neighbors = neighbors[:, 0]
                y_neighbors = neighbors[:, 1]
                z_neighbors = neighbors[:, 2]
                
                # Fit polynomial surface z = f(x, y)
                coeffs = self._fit_polynomial_surface(x_neighbors, y_neighbors, z_neighbors, order)
                
                # Step 5: Project point onto surface and calculate normal
                if coeffs is not None:
                    smoothed_point, normal = self._project_point_to_surface(point, coeffs, order)
                    smoothed_points.append(smoothed_point)
                    surface_normals.append(normal)
                else:
                    # Fallback if surface fitting fails
                    smoothed_points.append(point)
                    surface_normals.append([0, 0, 1])
        
        return np.array(smoothed_points), np.array(surface_normals), cleaned_pixels
    
    def _fit_polynomial_surface(self, x, y, z, order):
        n_points = len(x)
        
        if order == 1:
            # Linear surface: z = a0 + a1*x + a2*y
            A = np.column_stack([np.ones(n_points), x, y])
        elif order == 2:
            # Quadratic surface: z = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y²
            A = np.column_stack([np.ones(n_points), x, y, x*x, x*y, y*y])
        else:
            return None
        
        # Check matrix condition for numerical stability
        if np.linalg.cond(A) > 1e12:
            return None
        
        # Solve least squares system
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        
        return coeffs if rank == A.shape[1] else None
    
    def _project_point_to_surface(self, point, coeffs, order):
    	# Initialization
        x, y, z = point
        
        if order == 1:
            # Linear surface
            a0, a1, a2 = coeffs
            z_fitted = a0 + a1 * x + a2 * y
            normal = np.array([-a1, -a2, 1.0])
            
        elif order == 2:
            # Quadratic surface
            a0, a1, a2, a3, a4, a5 = coeffs
            z_fitted = a0 + a1*x + a2*y + a3*x*x + a4*x*y + a5*y*y
            
            # Calculate gradients for normal vector
            dfdx = a1 + 2*a3*x + a4*y  # ∂f/∂x
            dfdy = a2 + a4*x + 2*a5*y  # ∂f/∂y
            normal = np.array([-dfdx, -dfdy, 1.0])
        
        # Normalize the normal vector
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude > 1e-6:
            normal = normal / normal_magnitude
        else:
            normal = np.array([0, 0, 1])  # Default upward normal
        
        smoothed_point = np.array([x, y, z_fitted])
        
        return smoothed_point, normal
    
    def complete_preprocessing_pipeline(self, depth_map, fx, fy, cx, cy, x_min, x_max, y_min, y_max, z_min, z_max, m=50, k=1.0, search_radius=0.0005, order=2, verbose=True):
        if verbose:
            print("=== STARTING PREPROCESSING PIPELINE ===")
        
        # Step 1: Point Cloud Generation
        if verbose:
            print("Step 1/4: Point Cloud Generation")
        array_3d, valid_pixels = self.point_cloud_generation(depth_map, fx, fy, cx, cy)
        if verbose:
            print(f"Generated {len(array_3d):,} initial points")
        
        # Step 2: PassThrough Filtering
        if verbose:
            print("Step 2/4: PassThrough Filtering")
        filtered_points, filtered_pixels = self.pass_through_filter(array_3d, valid_pixels, x_min, x_max, y_min, y_max, z_min, z_max)
        if verbose:
            print(f"Filtered to {len(filtered_points):,} points")
        
        # Step 3: Statistical Outlier Removal
        if verbose:
            print("Step 3/4: Statistical Outlier Removal")
        cleaned_points, cleaned_pixels = self.statistical_outlier_removal(filtered_points, filtered_pixels, m, k)
        if verbose:
            print(f"Cleaned to {len(cleaned_points):,} points")
        
        # Step 4: Surface Smoothing MLS
        if verbose:
            print("Step 4/4: Surface Smoothing MLS")
        smoothed_points, surface_normals, smoothed_pixels = self.surface_smoothing_mls(cleaned_points, cleaned_pixels, search_radius, order)
        if verbose:
            print(f"Smoothed {len(smoothed_points):,} final points with normals")
            print("=== PREPROCESSING PIPELINE COMPLETE ===")
            print(f"Final result: {len(smoothed_points):,} smoothed points with surface normals")
        
        return smoothed_points, surface_normals, smoothed_pixels
    
    def analyze_point_cloud(self, array_3d):
        if len(array_3d) == 0:
            print("No points to analyze!")
            return {}
        
        analysis_results = {}
        
        print(f"\n=== DETAILED POINT CLOUD ANALYSIS ===")
        print(f"Total points: {len(array_3d):,}")
        
        # Statistical analysis for each axis
        for i, axis in enumerate(['X', 'Y', 'Z']):
            data = array_3d[:, i]
            axis_stats = {
                'min': data.min(),
                'max': data.max(),
                'span': data.max() - data.min(),
                'mean': data.mean(),
                'std': data.std()
            }
            analysis_results[axis] = axis_stats
            
            print(f"\n{axis}-axis statistics:")
            print(f"  Range: {axis_stats['min']:.3f}m to {axis_stats['max']:.3f}m")
            print(f"  Span: {axis_stats['span']:.3f}m ({axis_stats['span']*100:.1f}cm)")
            print(f"  Mean: {axis_stats['mean']:.3f}m")
            print(f"  Std Dev: {axis_stats['std']:.3f}m")
        
        # Density analysis
        volume = (analysis_results['X']['span'] * 
                 analysis_results['Y']['span'] * 
                 analysis_results['Z']['span'])
        
        if volume > 0:
            density = len(array_3d) / volume
            analysis_results['density'] = density
            print(f"\nPoint cloud density: {density:.0f} points/m³")
        
        # Distance analysis from origin
        distances = np.sqrt(np.sum(array_3d**2, axis=1))
        distance_stats = {
            'min_distance': distances.min(),
            'max_distance': distances.max(),
            'avg_distance': distances.mean()
        }
        analysis_results['distances'] = distance_stats
        
        print(f"\nDistance from camera origin:")
        print(f"  Closest point: {distance_stats['min_distance']:.3f}m")
        print(f"  Farthest point: {distance_stats['max_distance']:.3f}m")
        print(f"  Average distance: {distance_stats['avg_distance']:.3f}m")
        
        return analysis_results
