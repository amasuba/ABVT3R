import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

class ThreeDReconstruction:
    """
    A 3D reconstruction pipline implementation class from first principles

    This class will work to implement 3D Reconstruction using the following approaches:
    - Point Cloud Merging: Enhanced version of exisiting approach
    - Mesh Generation: generation of the plant mesh using triangulation techniques
    - Hole Filling: Finding and filling the holes 
    - Surface reconstruction: Enhanced surface smoothing
    """

    def __init__(self, verbose = True):
        self.verbose = verbose
        
    def log(self, message):
        if self.verbose:
            print(message)
            
    # ========================================================================
    # 1. Point Cloud Merging - Enhanced version of exisiting approach
    # ========================================================================

    def merge_registered_point_clouds(self, fine_registered_pcs):
        """
        Integrate the 4 registered point clouds into a single coherent 3D model
        This replaces the previous implementation
        """
        self.log("=" * 60)
        self.log("POINT CLOUD MERGING - FIRST PRINCIPLES")
        self.log("=" * 60)
        
        angles = [0, 90, 180, 270]
        point_counts = []
        all_points = []
        view_labels = []
        
        # Combine all point clouds
        for i, pc in enumerate(fine_registered_pcs):
            point_counts.append(len(pc))
            all_points.append(pc)
            
            # Create view labels for this point cloud
            labels = np.full(len(pc), i, dtype = int)
            view_labels.append(labels)
            
            self.log(f"View {i} ({angles[i]}degrees): {len(pc):,} points")
            
        # Stack into single array
        merged_cloud = np.vstack(all_points)
        view_labels = np.concatenate(view_labels)
        total_points = len(merged_cloud)
        
        # Calcualte quality metrics
        # 1. Coverage balance
        min_points = min(point_counts)
        max_points = max(point_counts)
        coverage_balance = min_points / max_points if max_points > 0 else 0
        
        # 2. Spatial extent
        x_range = merged_cloud[:, 0].max() - merged_cloud[:, 0].min()
        y_range = merged_cloud[:, 1].max() - merged_cloud[:, 1].min()
        z_range = merged_cloud[:, 2].max() - merged_cloud[:, 2].min()
        
        # 3. Point density uniformity
        density_uniformity = self.calculate_point_density_uniformity(merged_cloud)
        
        # 4. View overlap assessment
        overlap_scores = []
        for i in range(len(fine_registered_pcs)):
            for j in range(i + 1, len(fine_registered_pcs)):
                overlap = self.calculate_overlap(fine_registered_pcs[i], fine_registered_pcs[j])
                overlap_scores.append(overlap)
                
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        
        merge_quality = {
            'total_points': total_points,
            'point_counts': point_counts,
            'coverage_balance': coverage_balance,
            'spatial_extent': {'x': x_range, 'y': y_range, 'z': z_range},
            'density_uniformity': density_uniformity,
            'average_overlap': avg_overlap,
            'infividual_overlaps': overlap_scores
        }
        
        self.log(f"Successfully merged {total_points:,} points")
        self.log(f"Coverage balance: {coverage_balance: .3f}")
        self.log(f"Point density uniformity: {density_uniformity: .3f}")
        self.log(f"Average view overlap: {avg_overlap: .3f}")
        
        # Voxel downsampling
        voxel_size = 0.005 # 5mm voxels
        grid_coords = np.floor(merged_cloud / voxel_size).astype(int)
        unique_coords, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
        merged_cloud = merged_cloud[unique_indices]
        view_labels = view_labels[unique_indices]

        print(f"Downsampled to {len(merged_cloud):,} points")

        return merged_cloud, view_labels, merge_quality
        
    def calculate_point_density_uniformity(self, points, grid_size = 0.02):
        """
        Calculate spatial uniformity of point distribution
        Divides space into grid and measures density variation
        """
        # Create 3D grid
        min_coords = np.min(points, axis = 0)
        max_coords = np.max(points, axis = 0)
        
        grid_dims = np.ceil((max_coords - min_coords) / grid_size).astype(int)
        
        # Assign points to grid cells
        grid_indices = np.floor((points - min_coords) / grid_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_dims - 1)
        
        # Count points per cell
        grid_counts = {}
        for index in grid_indices:
            key = tuple(index)
            grid_counts[key] = grid_counts.get(key, 0) + 1
        
        # calcualte density uniformity
        counts = list(grid_counts.values())
        if len(counts) == 0:
            return 0.0
        
        # Use coefficient of variation (CV) for better metric
        mean_density = np.mean(counts)
        std_density = np.std(counts)
        
        if mean_density == 0:
            return 0.0
        
        cv = std_density / mean_density
        # Convert to uniformity score: CV of 0 = perfect uniformity (1.0)
        # CV of 2 or higher = very non-uniform (0.0)
        uniformity = max(0, 1 - (cv / 2))
        
        return uniformity
        
    def calculate_overlap(self, pc1, pc2, threshold = 0.015):
        """
        Calculate overlap between to point clouds using nearest neighbor search
        """
        if len(pc1) == 0 or len(pc2) == 0:
            return 0.0
            
        # Sample for efficiency
        sample_size = min(1000, len(pc1))
        sample_indices = np.random.choice(len(pc1), sample_size, replace = False)
        pc1_sample = pc1[sample_indices]
        
        # Use NearestNeighbors for efficiency
        tree = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(pc2)
        
        distances, _ = tree.kneighbors(pc1_sample)
        overlap_count = np.sum(distances.flatten() < threshold)
        
        return overlap_count / len(pc1_sample)
        
    # =====================================================================
    # 2. Mesh Generation
    # =====================================================================
    def repair_non_manifold_mesh(self, vertices, triangles):
        """Remove non-manifold edges and degenerate triangles"""
        edge_count = {}
        
        for tri_idx, triangle in enumerate(triangles):
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]
            
            for edge in edges:
                if edge not in edge_count:
                    edge_count[edge] = []
                edge_count[edge].append(tri_idx)
        
        # Find triangles with non-manifold edges (>2 triangles share edge)
        bad_triangle_indices = set()
        for edge, tri_indices in edge_count.items():
            if len(tri_indices) > 2:
                # Keep first 2, mark others as bad
                bad_triangle_indices.update(tri_indices[2:])
        
        # Keep only manifold triangles
        clean_triangles = np.array([tri for idx, tri in enumerate(triangles) 
                                    if idx not in bad_triangle_indices])
        
        return vertices, clean_triangles
        
    def generate_plant_mesh(self, merged_cloud, method='projection', **params):
        """
        Create geometric mesh representation - FAST NUMPY-ONLY methods
        
        Methods:
        - 'projection': Multi-view 2.5D projection (RECOMMENDED - Very Fast)
        - 'greedy_projection': Greedy triangulation projection (Fast)
        - 'grid_based': Voxel grid surface extraction (Very Fast)
        """
        self.log("=" * 60)
        self.log(f"MESH GENERATION - {method.upper()}")
        self.log("=" * 60)
        
        if method == 'projection':
            vertices, triangles = self.multiview_projection_mesh(
                merged_cloud, 
                grid_resolution=params.get('grid_resolution', 0.005)
            )
        elif method == 'greedy_projection':
            vertices, triangles = self.greedy_projection_triangulation(
                merged_cloud,
                max_edge_length=params.get('max_edge_length', 0.03)
            )
        elif method == 'grid_based':
            vertices, triangles, volume_data = self.grid_based_surface_mesh(
                merged_cloud,
                voxel_size=params.get('voxel_size', 0.01)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Repair non-manifold edges
        vertices, triangles = self.repair_non_manifold_mesh(vertices, triangles)
        
        # Calculate mesh quality
        mesh_quality = self.assess_mesh_quality(vertices, triangles, merged_cloud)
        
        self.log(f"Generated mesh: {len(vertices):,} vertices, {len(triangles):,} triangles")
        self.log(f"Mesh quality score: {mesh_quality['quality_score']:.3f}")
        
        if method == 'grid_based':
            return vertices, triangles, mesh_quality, volume_data
        else:
            return vertices, triangles, mesh_quality
        
    # ===========================================================================
    # METHOD 1: MULTI-VIEW PROJECTION (RECOMMENDED FOR PLANTS)
    # ===========================================================================
    # Time: O(n log n) - Very fast
    # Perfect for plants captured from multiple views (your use case!)

    def multiview_projection_mesh(self, points, grid_resolution=0.005):
        """
        Multi-view 2.5D projection triangulation
        - Projects points onto multiple planes (top, front, side)
        - Creates 2D triangulation on each plane (Delaunay-like, but 2D is fast!)
        - Combines meshes from all views
        """
        self.log(f"Multi-view projection with resolution {grid_resolution}m...")
        
        all_triangles = []
        
        # Define projection planes (matching your 4-view capture system)
        projections = [
            ('XY', [0, 1], 2),  # Top view (project to XY plane)
            ('XZ', [0, 2], 1),  # Front view (project to XZ plane)
            ('YZ', [1, 2], 0),  # Side view (project to YZ plane)
        ]
        
        for proj_name, plane_dims, depth_dim in projections:
            self.log(f"  Processing {proj_name} projection...")
            
            # Project points onto 2D plane
            points_2d = points[:, plane_dims]
            depths = points[:, depth_dim]
            
            # Create 2D grid-based triangulation (FAST!)
            triangles_2d = self._fast_2d_grid_triangulation(
                points_2d, depths, grid_resolution
            )
            
            all_triangles.extend(triangles_2d)
            self.log(f"    Generated {len(triangles_2d)} triangles from {proj_name}")
        
        # Remove duplicate triangles
        unique_triangles = self._remove_duplicate_triangles(all_triangles)
        
        self.log(f"Combined {len(unique_triangles)} unique triangles from all views")
        
        return points, np.array(unique_triangles, dtype=int)

    def _fast_2d_grid_triangulation(self, points_2d, depths, grid_res):
        """
        Fast 2D triangulation using structured grid
        This is like 2D Delaunay but MUCH faster using grid structure
        """
        # Create 2D grid
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        
        grid_dims = np.ceil((max_coords - min_coords) / grid_res).astype(int) + 1
        
        # Assign points to grid cells
        grid_indices = np.floor((points_2d - min_coords) / grid_res).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_dims - 1)
        
        # Create grid structure (cell -> point index)
        grid = {}
        for point_idx in range(len(points_2d)):
            cell = tuple(grid_indices[point_idx])
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(point_idx)
        
        # Create triangles by connecting adjacent grid cells
        triangles = []
        
        for i in range(grid_dims[0] - 1):
            for j in range(grid_dims[1] - 1):
                # Get points in 2x2 cell neighborhood
                cells = [
                    (i, j), (i+1, j), (i+1, j+1), (i, j+1)
                ]
                
                cell_points = []
                for cell in cells:
                    if cell in grid:
                        cell_points.extend(grid[cell])
                
                if len(cell_points) >= 3:
                    # Create triangles from cell points
                    # Use point closest to each corner
                    corners = self._select_corner_points(
                        cell_points, cells, points_2d, min_coords, grid_res
                    )
                    
                    if len(corners) >= 3:
                        # Create two triangles for the quad
                        triangles.append([corners[0], corners[1], corners[2]])
                        if len(corners) >= 4:
                            triangles.append([corners[0], corners[2], corners[3]])
        
        return triangles

    def _select_corner_points(self, point_indices, cells, points_2d, origin, grid_res):
        """Select best point for each corner of the grid cell"""
        corners = []
        
        for cell in cells:
            cell_center = origin + (np.array(cell) + 0.5) * grid_res
            
            # Find closest point to this corner
            best_idx = None
            min_dist = float('inf')
            
            for idx in point_indices:
                dist = np.linalg.norm(points_2d[idx] - cell_center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            
            if best_idx is not None and best_idx not in corners:
                corners.append(best_idx)
        
        return corners

    def _remove_duplicate_triangles(self, triangles):
        """Remove duplicate triangles using sorted tuples"""
        unique = set()
        result = []
        
        for tri in triangles:
            # Create canonical form (sorted indices)
            key = tuple(sorted(tri))
            if key not in unique:
                unique.add(key)
                result.append(tri)
        
        return result

    # ===========================================================================
    # METHOD 2: GREEDY PROJECTION TRIANGULATION
    # ===========================================================================
    # Time: O(n log n) - Fast
    # Simple greedy approach that works well for surfaces

    def greedy_projection_triangulation(self, points, max_edge_length=0.03):
        """
        Greedy projection triangulation - NumPy only
        Builds mesh by greedily connecting nearby points
        """
        self.log(f"Greedy projection triangulation (max edge: {max_edge_length}m)...")
        
        n_points = len(points)
        used_points = set()
        triangles = []
        
        # Start with centroid-nearest point
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        start_idx = np.argmin(distances)
        
        # Build mesh by expanding from seed
        active_edges = []
        
        # Find first triangle
        seed_tri = self._find_seed_triangle_greedy(points, start_idx, max_edge_length)
        if seed_tri is not None:
            triangles.append(seed_tri)
            used_points.update(seed_tri)
            
            # Add edges to active list
            active_edges.extend([
                (seed_tri[0], seed_tri[1]),
                (seed_tri[1], seed_tri[2]),
                (seed_tri[2], seed_tri[0])
            ])
        
        # Greedy expansion
        iterations = 0
        max_iterations = n_points * 2
        
        while active_edges and iterations < max_iterations:
            iterations += 1
            
            if iterations % 500 == 0:
                progress = (len(used_points) / n_points) * 100
                self.log(f"  Progress: {progress:.0f}% ({len(triangles)} triangles)")
            
            # Get an edge
            edge = active_edges.pop(0)
            p1_idx, p2_idx = edge
            
            # Find best point to complete triangle
            candidate = self._find_greedy_candidate(
                points, p1_idx, p2_idx, max_edge_length, used_points
            )
            
            if candidate is not None:
                new_tri = [p1_idx, p2_idx, candidate]
                triangles.append(new_tri)
                used_points.add(candidate)
                
                # Add new edges
                new_edges = [
                    (p2_idx, candidate),
                    (candidate, p1_idx)
                ]
                
                for new_edge in new_edges:
                    # Check if edge is not already used
                    if not self._edge_exists(new_edge, triangles):
                        active_edges.append(new_edge)
        
        self.log(f"Greedy triangulation: {len(triangles)} triangles, {len(used_points)} points used")
        
        return points, np.array(triangles, dtype=int)

    def _find_seed_triangle_greedy(self, points, start_idx, max_edge):
        """Find initial triangle for greedy expansion"""
        p1 = points[start_idx]
        
        # Find nearby points
        distances = np.linalg.norm(points - p1, axis=1)
        nearby = np.where((distances > 0) & (distances < max_edge))[0]
        
        if len(nearby) < 2:
            return None
        
        # Try first few combinations
        for i in range(min(5, len(nearby))):
            for j in range(i + 1, min(10, len(nearby))):
                p2_idx = nearby[i]
                p3_idx = nearby[j]
                
                # Check triangle quality
                tri_points = points[[start_idx, p2_idx, p3_idx]]
                if self._is_good_triangle(tri_points, max_edge):
                    return [start_idx, p2_idx, p3_idx]
        
        return None

    def _find_greedy_candidate(self, points, p1_idx, p2_idx, max_edge, used):
        """Find best candidate point for greedy triangulation"""
        p1, p2 = points[p1_idx], points[p2_idx]
        edge_midpoint = (p1 + p2) / 2
        
        # Find candidates near edge
        distances = np.linalg.norm(points - edge_midpoint, axis=1)
        candidates = np.where(distances < max_edge)[0]
        
        best_candidate = None
        best_score = float('inf')
        
        for candidate_idx in candidates:
            if candidate_idx in used or candidate_idx == p1_idx or candidate_idx == p2_idx:
                continue
            
            # Score based on distance and angle
            tri_points = points[[p1_idx, p2_idx, candidate_idx]]
            
            if self._is_good_triangle(tri_points, max_edge):
                # Prefer points that create regular triangles
                score = self._triangle_regularity_score(tri_points)
                
                if score < best_score:
                    best_score = score
                    best_candidate = candidate_idx
        
        return best_candidate

    def _is_good_triangle(self, tri_points, max_edge):
        """Check if triangle has reasonable proportions"""
        p1, p2, p3 = tri_points
        
        # Check edge lengths
        edges = [
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p3 - p2),
            np.linalg.norm(p1 - p3)
        ]
        
        # All edges should be reasonable
        if max(edges) > max_edge or min(edges) < 0.001:
            return False
        
        # Check area (avoid degenerate triangles)
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        return area > 1e-6

    def _triangle_regularity_score(self, tri_points):
        """Score triangle regularity (lower is better)"""
        p1, p2, p3 = tri_points
        
        edges = [
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p3 - p2),
            np.linalg.norm(p1 - p3)
        ]
        
        # Ratio of longest to shortest edge
        regularity = max(edges) / (min(edges) + 1e-10)
        
        return regularity

    def _edge_exists(self, edge, triangles):
        """Check if edge exists in triangle list"""
        e1, e2 = edge
        
        for tri in triangles[-10:]:  # Check only recent triangles for speed
            if (e1 in tri and e2 in tri):
                return True
        
        return False

    # ===========================================================================
    # METHOD 3: GRID-BASED SURFACE MESH
    # ===========================================================================
    # Time: O(n) - VERY FAST!
    # Creates mesh from voxel grid surface

    def grid_based_surface_mesh(self, points, voxel_size=0.01):
        """
        Ultra-fast grid-based surface mesh
        Creates mesh from occupied voxel boundaries
        """
        self.log(f"Grid-based mesh generation (voxel: {voxel_size}m)...")
        
        # Create voxel grid
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 2
        self.log(f"Grid dimensions: {grid_dims}")
        
        # Mark occupied voxels
        grid = np.zeros(grid_dims, dtype=bool)
        
        for point in points:
            idx = np.floor((point - min_coords) / voxel_size).astype(int) + 1
            idx = np.clip(idx, 0, grid_dims - 1)
            grid[tuple(idx)] = True
        
        # Extract surface voxels (have at least one empty neighbor)
        self.log("Extracting surface voxels...")
        vertices = []
        triangles = []
        vertex_map = {}
        
        for i in range(1, grid_dims[0] - 1):
            if i % 10 == 0:
                progress = (i / grid_dims[0]) * 100
                self.log(f"  Progress: {progress:.0f}%")
            
            for j in range(1, grid_dims[1] - 1):
                for k in range(1, grid_dims[2] - 1):
                    if not grid[i, j, k]:
                        continue
                    
                    # Check if surface voxel (has empty neighbor)
                    neighbors = [
                        grid[i-1, j, k], grid[i+1, j, k],
                        grid[i, j-1, k], grid[i, j+1, k],
                        grid[i, j, k-1], grid[i, j, k+1]
                    ]
                    
                    if not all(neighbors):
                        # Create cube faces for this surface voxel
                        cube_triangles = self._create_voxel_surface(
                            i, j, k, grid, min_coords, voxel_size, vertex_map, vertices
                        )
                        triangles.extend(cube_triangles)
        
        self.log(f"Grid-based mesh: {len(vertices)} vertices, {len(triangles)} triangles")
        
        # Volume data calculation
        volume_data = (grid, voxel_size)
        
        return np.array(vertices), np.array(triangles, dtype=int), volume_data

    def _create_voxel_surface(self, i, j, k, grid, origin, voxel_size, vertex_map, vertices):
        """Create triangular faces for exposed voxel faces"""
        triangles = []
        
        # Define 6 faces of cube and their 4 corners
        faces = [
            # Face, normal direction, 4 corners
            ('front', [0, 0, 1], [(0,0,1), (1,0,1), (1,1,1), (0,1,1)]),
            ('back',  [0, 0,-1], [(0,0,0), (0,1,0), (1,1,0), (1,0,0)]),
            ('right', [1, 0, 0], [(1,0,0), (1,1,0), (1,1,1), (1,0,1)]),
            ('left',  [-1,0, 0], [(0,0,0), (0,0,1), (0,1,1), (0,1,0)]),
            ('top',   [0, 1, 0], [(0,1,0), (0,1,1), (1,1,1), (1,1,0)]),
            ('bottom',[0,-1, 0], [(0,0,0), (1,0,0), (1,0,1), (0,0,1)]),
        ]
        
        for face_name, normal, corners in faces:
            # Check if this face is exposed (neighbor is empty)
            ni, nj, nk = i + normal[0], j + normal[1], k + normal[2]
            
            if not grid[ni, nj, nk]:
                # This face is exposed - create two triangles
                corner_indices = []
                
                for corner in corners:
                    # Get or create vertex
                    world_pos = origin + voxel_size * (np.array([i, j, k]) + np.array(corner))
                    vertex_key = tuple(world_pos)
                    
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = len(vertices)
                        vertices.append(world_pos)
                    
                    corner_indices.append(vertex_map[vertex_key])
                
                # Create two triangles from quad
                triangles.append([corner_indices[0], corner_indices[1], corner_indices[2]])
                triangles.append([corner_indices[0], corner_indices[2], corner_indices[3]])
        
        return triangles
        
    # ======================================================================
    # 3. Hole Filling 
    # ======================================================================
    
    def detect_and_fill_holes(self, vertices, triangles, max_hole_size = 0.01):
        """
        Address any gaps or missing data in areas between the 4 angles
        """
        self.log("=" * 60)
        self.log("HOLE DETECTION AND FILLING")
        self.log("=" * 60)
        
        # Step 1: Find boundary edges 
        boundary_edges = self.find_boundary_edges(triangles)
        self.log(f"Found {len(boundary_edges)} boundary edges")
        
        # Step 2: Group boundary edges into holes
        holes = self.group_edges_into_holes(boundary_edges, vertices, max_hole_size)
        self.log(f"Detected {len(holes)} holes to fill")
        
        # Step 3: Fill holes using triangulation
        filled_vertices = vertices.copy()
        filled_triangles = triangles.copy()
        
        holes_filled = 0
        vertices_added = 0
        triangles_added = 0
        
        for hole in holes:
            if hole['size'] <= max_hole_size:
                new_verts, new_tris = self.fill_hole_with_triangulation(filled_vertices, hole, len(filled_vertices))
                
                filled_vertices = np.vstack([filled_vertices, new_verts])
                filled_triangles = np.vstack([filled_triangles, new_tris])
                
                holes_filled += 1
                vertices_added += len(new_verts)
                triangles_added += len(new_tris)
                
                self.log(f"    Filled hole {holes_filled}: +{len(new_verts)} vertices, +{len(new_tris)} triangles")
                
        hole_info = {
            'holes_detected': len(holes),
            'holes_filled': holes_filled,
            'vertices_added': vertices_added,
            'triangles_added': triangles_added
        }
        
        self.log(f"Hole filling complete: {holes_filled}/{len(holes)} holes filled")
        
        return filled_vertices, filled_triangles, hole_info
        
    def find_boundary_edges(self, triangles):
        """
        Find edges that belong to only one triangle
        """
        edge_count = {}
        for triangle in triangles:
            # Get three edges of triangle
            edges = [tuple(sorted([triangle[0], triangle[1]])), tuple(sorted([triangle[1], triangle[2]])), tuple(sorted([triangle[2], triangle[0]]))]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
                
        # Boundary edges appear exactly once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return boundary_edges
        
    def group_edges_into_holes(self, boundary_edges, vertices, max_hole_size):
        """
        Group connected boundary edges into holes using graph traversal
        """
        # Build adjacency graph
        adjacency = {}
        for edge in boundary_edges:
            vertex1, vertex2 = edge
            if vertex1 not in adjacency:
                adjacency[vertex1] = []
            if vertex2 not in adjacency:
                adjacency[vertex2] = []
            
            adjacency[vertex1].append(vertex2)
            adjacency[vertex2].append(vertex1)
            
        # Find connected components
        visited = set()
        holes = []
        
        for start_vertex in adjacency:
            if start_vertex not in visited:
                # Traverse connected component
                hole_vertices = []
                queue = [start_vertex]
                visited.add(start_vertex)
                
                while queue:
                    current = queue.pop(0)
                    hole_vertices.append(current)
                    
                    for neighbor in adjacency.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            
                # Analyze this potential hole
                if len(hole_vertices) >= 3:
                    hole_points = vertices[hole_vertices]
                    hole_size = self.calculate_hole_size(hole_points)
                    hole_center = np.mean(hole_points, axis = 0)
                    
                    holes.append({
                        'vertices': hole_vertices,
                        'size': hole_size,
                        'center': hole_center,
                        'perimeter': len(hole_vertices)
                    })
                    
        return holes
        
    def calculate_hole_size(self, hole_points):
        """
        Calculate hole size as maximum distance between boundary points
        """
        max_distance = 0
        for i in range(len(hole_points)):
            for j in range(i + 1, len(hole_points)):
                distance = np.linalg.norm(hole_points[i] - hole_points[j])
                if distance > max_distance:
                    max_distance = distance
                    
        return max_distance
        
    def fill_hole_with_triangulation(self, vertices, hole, start_vertex_index):
        """
        Fill hole using center point and fan triangulation
        """
        hole_vertices = hole['vertices']
        hole_center = hole['center']
        
        # Add center point
        new_vertices = np.array([hole_center])
        center_index = start_vertex_index
        
        # Order boundary vertices in circular fashion
        hole_points = vertices[hole_vertices]
        
        # Calculate angles around center for ordering
        vectors_to_center = hole_points - hole_center
        angles = np.arctan2(vectors_to_center[:, 1], vectors_to_center[:, 0])
        sorted_indices = np.argsort(angles)
        ordered_vertices = [hole_vertices[index] for index in sorted_indices]
        
        # Create fan triangulation
        new_triangles = []
        for i in range(len(ordered_vertices)):
            vertex1 = ordered_vertices[i]
            vertex2 = ordered_vertices[(i + 1) % len(ordered_vertices)]
            
            # Ensure consisten winding order
            new_triangles.append([vertex1, vertex2, center_index])
            
        return new_vertices, np.array(new_triangles, dtype = int)
        
    # ===============================================================
    # 4. Surface Reconstruction
    # ===============================================================
    
    def reconstruct_smooth_surface(self, vertices, triangles, iterations = 5, preserve_features = True):
        """
        Generate smooth surfaces whle preserving important plant features
        Uses Laplacian smoothing
        """
        self.log("=" * 60)
        self.log("SURFACE RECONSTRUCTION")
        self.log("=" * 60)
        
        # Build vertex adjacency
        vertex_adjacency = self.build_vertex_adjacency(triangles, len(vertices))
        
        smoothed_vertices = vertices.copy()
        
        # Iterative Laplacian smoothing
        for iteration in range(iterations):
            self.log(f"Smoothing iteration {iteration + 1}/{iterations}")
            
            new_vertices = smoothed_vertices.copy()
            
            # Progressive smoothing: start gentle and increase strength
            base_factor = 0.3 + (0.4 * iteration / iterations)
            
            for i in range(len(vertices)):
                neighbors = vertex_adjacency.get(i, [])
                
                if len(neighbors) > 0:
                    # Calculate laplacian 
                    neighbor_positions = smoothed_vertices[neighbors]
                    mean_neighbor = np.mean(neighbor_positions, axis = 0)
                    laplacian = mean_neighbor - smoothed_vertices[i]
                    
                    # Apply smoothing with damping
                    smoothing_factor = base_factor
                    if preserve_features:
                        # Reduce smoothing near sharp features
                        feature_strength = self.calculate_feature_strength(i, neighbors, smoothed_vertices)
                        smoothing_factor *= (1 - feature_strength * 0.7)
                        
                    new_vertices[i] = smoothed_vertices[i] + smoothing_factor * laplacian
                    
            smoothed_vertices = new_vertices
            
        # Calculate surface normals
        surface_normals = self.calculate_vertex_normals(smoothed_vertices, triangles)
        
        # Assess surface quality
        surface_quality = self.assess_surface_quality(smoothed_vertices, triangles, vertices)
        
        self.log(f"Surface reconstruction complete")
        self.log(f"Feature preservation: {'ON' if preserve_features else 'OFF'}")
        self.log(f"Surface quality score: {surface_quality['quality_score']: .3f}")
        
        return smoothed_vertices, surface_normals, surface_quality
        
    def build_vertex_adjacency(self, triangles, num_vertices):
        """
        Build vertex adjacency list from triangles
        """
        adjacency = {}
        
        for triangle in triangles:
            for i in range(3):
                vertex1 = triangle[i]
                vertex2 = triangle[(i + 1) % 3]
                vertex3 = triangle[(i + 2) % 3]
                
                if vertex1 not in adjacency:
                    adjacency[vertex1] = set()
                    
                adjacency[vertex1].add(vertex2)
                adjacency[vertex1].add(vertex3)
                
        # Convert sets to lists
        for vertex in adjacency:
            adjacency[vertex] = list(adjacency[vertex])
            
        return adjacency
        
    def calculate_feature_strength(self, vertex_index, neighbors, vertices):
        """
        Calculate how much this vertex represents a sharp feature
        Higher values = sharper feature == less smoothing
        """
        if len(neighbors) < 3:
            return 1.0 # Boundary vertices are features
            
        vertex = vertices[vertex_index]
        neighbor_positions = vertices[neighbors]
        
        # Calculate angles between edges from this vertex
        angles = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                vector1 = neighbor_positions[i] - vertex
                vector2 = neighbor_positions[j] - vertex
                
                # Normalize vectors
                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)
                
                if norm1 > 0 and norm2 > 0:
                    vector1_unit = vector1 / norm1
                    vector2_unit = vector2 / norm2
                    
                    # Calculate angle
                    cos_angle = np.clip(np.dot(vector1_unit, vector2_unit), -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                    
        if len(angles) == 0:
            return 0.0
            
        # Sharp features have small angles
        min_angle = np.min(angles)
        feature_strength = max(0, (np.pi / 3 - min_angle) / (np.pi / 3)) # Sharp if < 60°
        
        return feature_strength
        
    def calculate_vertex_normals(self, vertices, triangles):
        """
        Calculate vertex normals by averaging adjacent triangle normals
        """
        vertex_normals = np.zeros_like(vertices)
        vertex_counts = np.zeros(len(vertices))
        
        for triangle in triangles:
            vertex1, vertex2, vertex3 = triangle
            
            # Get triangle vertices
            p1, p2, p3 = vertices[vertex1], vertices[vertex2], vertices[vertex3]
            
            # Calculate triangle normal
            edge1 = p2 - p1
            edge2 = p3 - p1
            normal = np.cross(edge1, edge2)
            
            # Normalize
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
                
                # Add to vertex normals
                vertex_normals[vertex1] += normal
                vertex_normals[vertex2] += normal
                vertex_normals[vertex3] += normal
                vertex_counts[vertex1] += 1
                vertex_counts[vertex2] += 1
                vertex_counts[vertex3] += 1
                
        # Average normals
        for i in range(len(vertex_normals)):
            if vertex_counts[i] > 0:
                vertex_normals[i] /= vertex_counts[i]
                # Noemalize
                norm_length = np.linalg.norm(vertex_normals[i])
                if norm_length > 0:
                    vertex_normals[i] /= norm_length
                    
        return vertex_normals
        
    # ===============================================================
    # QUALITY ASSESSMENT
    # ===============================================================
    
    def assess_mesh_quality(self, vertices, triangles, original_points):
        """
        Assess mesh quality using first principles calculations
        """
        if len(vertices) == 0 or len(triangles) == 0:
            return {'quality_score': 0, 'error': 'Empty mesh'}

        # 1. Vertex count quality
        vertex_efficiency = min(1.0, len(vertices) / (len(original_points) * 0.5))

        # 2. Triangle quality (aspect ratios)
        triangle_qualities = []
        for triangle in triangles[:min(1000, len(triangles))]:  # Sample for efficiency
            v1, v2, v3 = vertices[triangle]

            # Calculate edge lengths
            edge1 = np.linalg.norm(v2 - v1)
            edge2 = np.linalg.norm(v3 - v2)  
            edge3 = np.linalg.norm(v1 - v3)

            # Triangle quality (closer to equilateral = better)
            max_edge = max(edge1, edge2, edge3)
            min_edge = min(edge1, edge2, edge3)
            quality = min_edge / max_edge if max_edge > 0 else 0
            triangle_qualities.append(quality)

        avg_triangle_quality = np.mean(triangle_qualities) if triangle_qualities else 0

        # 3. Coverage assessment (how well mesh covers original points)
        coverage_quality = self._assess_mesh_coverage(vertices, original_points)

        # Overall quality score
        quality_score = (vertex_efficiency * 0.3 + avg_triangle_quality * 0.4 + coverage_quality * 0.3)

        result = {'quality_score': quality_score, 'vertex_efficiency': vertex_efficiency, 'triangle_quality': avg_triangle_quality, 'coverage_quality': coverage_quality, 'vertex_count': len(vertices), 'triangle_count': len(triangles)}
        
        return result

    def _assess_mesh_coverage(self, mesh_vertices, original_points, max_distance=0.01):
        """
        Assess how well the mesh covers the original points
        """
        if len(mesh_vertices) == 0:
            return 0

        # Sample original points for efficiency
        sample_size = min(2000, len(original_points))
        sample_indices = np.random.choice(len(original_points), sample_size, replace=False)
        sample_points = original_points[sample_indices]

        covered_count = 0

        for point in sample_points:
            # Find distance to closest mesh vertex
            distances = np.linalg.norm(mesh_vertices - point, axis=1)
            min_distance = np.min(distances)

            if min_distance <= max_distance:
                covered_count += 1

        coverage_ratio = covered_count / len(sample_points)
        return coverage_ratio

    def assess_surface_quality(self, vertices, triangles, original_vertices):
        """
        Assess surface quality after reconstruction
        """
        # 1. Smoothness assessment
        smoothness = self._calculate_surface_smoothness(vertices, triangles)

        # 2. Geometric fidelity
        geometric_fidelity = self._calculate_geometric_fidelity(vertices, original_vertices)

        # 3. Triangle regularity
        triangle_regularity = self._calculate_triangle_regularity(vertices, triangles)

        # Overall surface quality
        quality_score = (smoothness * 0.4 + geometric_fidelity * 0.4 + triangle_regularity * 0.2)

        return {
            'quality_score': quality_score,
            'smoothness': smoothness,
            'geometric_fidelity': geometric_fidelity,
            'triangle_regularity': triangle_regularity,
            'is_manifold': self._check_manifold_property(triangles)
        }

    def _calculate_surface_smoothness(self, vertices, triangles):
        """
        Calculate surface smoothness by analyzing dihedral angles
        """
        # Build edge-triangle adjacency
        edge_triangles = {}

        for tri_idx, triangle in enumerate(triangles):
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]

            for edge in edges:
                if edge not in edge_triangles:
                    edge_triangles[edge] = []
                edge_triangles[edge].append(tri_idx)

        # Calculate dihedral angles for internal edges
        dihedral_angles = []

        for edge, tri_indices in edge_triangles.items():
            if len(tri_indices) == 2:  # Internal edge
                # Get normals of adjacent triangles
                tri1, tri2 = tri_indices

                normal1 = self._calculate_triangle_normal(vertices[triangles[tri1]])
                normal2 = self._calculate_triangle_normal(vertices[triangles[tri2]])

                # Calculate dihedral angle
                cos_angle = np.clip(np.dot(normal1, normal2), -1, 1)
                angle = np.arccos(cos_angle)
                dihedral_angles.append(angle)

        if len(dihedral_angles) == 0:
            return 0.5

        # Smoothness = how close angles are to 180° (flat surface)
        target_angle = np.pi
        angle_deviations = [abs(angle - target_angle) for angle in dihedral_angles]
        avg_deviation = np.mean(angle_deviations)

        # Normalize (0 = very rough, 1 = very smooth)
        smoothness = max(0, 1 - (avg_deviation / (np.pi/2)))

        return smoothness

    def _calculate_triangle_normal(self, triangle_vertices):
        """Calculate normal for a single triangle"""
        v1, v2, v3 = triangle_vertices
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)

        norm_length = np.linalg.norm(normal)
        if norm_length > 0:
            normal = normal / norm_length

        return normal

    def _calculate_geometric_fidelity(self, new_vertices, original_vertices):
        """
        Calculate how well the new mesh preserves the original geometry
        """
        if len(original_vertices) == 0 or len(new_vertices) == 0:
            return 0

        # Sample for efficiency
        sample_size = min(1000, len(original_vertices))
        sample_indices = np.random.choice(len(original_vertices), sample_size, replace=False)
        sample_points = original_vertices[sample_indices]

        total_distance = 0

        for point in sample_points:
            # Find closest vertex in new mesh
            distances = np.linalg.norm(new_vertices - point, axis=1)
            min_distance = np.min(distances)
            total_distance += min_distance

        avg_distance = total_distance / len(sample_points)

        # Convert to fidelity score (closer = better)
        max_acceptable_distance = 0.02  # 2cm
        fidelity = max(0, 1 - (avg_distance / max_acceptable_distance))

        return fidelity

    def _calculate_triangle_regularity(self, vertices, triangles):
        """
        Calculate triangle regularity (how close triangles are to equilateral)
        """
        regularities = []

        sample_triangles = triangles[:min(1000, len(triangles))]  # Sample for efficiency

        for triangle in sample_triangles:
            v1, v2, v3 = vertices[triangle]

            # Calculate edge lengths
            edge1 = np.linalg.norm(v2 - v1)
            edge2 = np.linalg.norm(v3 - v2)
            edge3 = np.linalg.norm(v1 - v3)

            edges = [edge1, edge2, edge3]

            # Regularity = ratio of smallest to largest edge
            min_edge = min(edges)
            max_edge = max(edges)

            regularity = min_edge / max_edge if max_edge > 0 else 0
            regularities.append(regularity)

        return np.mean(regularities) if regularities else 0

    def _check_manifold_property(self, triangles):
        """
        Check if mesh is a proper manifold (each edge shared by at most 2 triangles)
        """
        edge_count = {}

        for triangle in triangles:
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]

            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Check if any edge is shared by more than 2 triangles
        non_manifold_edges = sum(1 for count in edge_count.values() if count > 2)

        return non_manifold_edges == 0

    # ==============================================================
    # COMPLETE PIPELINE
    # ==============================================================
    
    def complete_reconstruction_pipeline(self, fine_registered_pcs, method = 'grid_based', **params):
        """
        Complete 3D reconstruction pipeline implementing all FU4.4 componets
        """
        self.log("\n" + "=" * 80)
        self.log("COMPLETE 3D RECONSTRUCTION PIPELINE")
        self.log("=" * 80)
        
        results = {}
        
        # Step 1: Point Cloud Merging
        merged_cloud, view_labels, merge_quality = self.merge_registered_point_clouds(fine_registered_pcs)
        results['merged_cloud'] = merged_cloud
        results['view_labels'] = view_labels
        results['merge_quality'] = merge_quality
                
        # Step 2: Mesh Generation
        if method == 'grid_based':
            mesh_vertices, mesh_triangles, mesh_quality, volume_data = self.generate_plant_mesh(merged_cloud, method=method, **params)
            results['volume_data'] = volume_data  # STORE IT
        else:
            mesh_vertices, mesh_triangles, mesh_quality = self.generate_plant_mesh(merged_cloud, method=method, **params)
            results['volume_data'] = None
            
        results['mesh_vertices'] = mesh_vertices
        results['mesh_triangles'] = mesh_triangles
        results['mesh_quality'] = mesh_quality
        results['method'] = method
        
        # Step 3: Hole Filling
        if method != 'grid_based':
            filled_vertices, filled_triangles, hole_info = self.detect_and_fill_holes(mesh_vertices, mesh_triangles, max_hole_size = params.get('hole_threshold', 0.01))
        else:
            self.log("Skipping hole filling (grid-based mesh is already complete)")
            filled_vertices = mesh_vertices
            filled_triangles = mesh_triangles
            hole_info = {
                'holes_detected': 0,
                'holes_filled': 0,
                'vertices_added': 0,
                'triangles_added': 0
            }
        
        results['filled_vertices'] = filled_vertices
        results['filled_triangles'] = filled_triangles
        results['hole_info'] = hole_info
        
        # Step 4: Surface Reconstruction
        final_vertices, surface_normals, surface_quality = self.reconstruct_smooth_surface(filled_vertices, filled_triangles, iterations = params.get('smooth_iterations', 5), preserve_features = params.get('preserve_features', True))
        results['final_vertices'] = final_vertices
        results['final_triangles'] = filled_triangles
        results['surface_normals'] = surface_normals
        results['surface_quality'] = surface_quality
        
        # Calculate overall reconstruction statistics
        reconstruction_stats = self.calculate_reconstruction_statistics(results, fine_registered_pcs)
        results['reconstruction_stats'] = reconstruction_stats
        
        self.log("\n" + "=" * 80)
        self.log("3D RECONSTRUCTION COMPLETE - SUMMARY")
        self.log("=" * 80)
        self.log(f"Input: {reconstruction_stats['input_points']:,} points from 4 views")
        self.log(f"Final: {reconstruction_stats['final_vertices']:,} vertices, {reconstruction_stats['final_triangles']:,} triangles")
        self.log(f"Holes filled:{reconstruction_stats['holes_filled']}")
        self.log(f"Overall quality: {reconstruction_stats['overall_quality']:.3f}")
        self.log("=" * 80)
        
        return results
        
    def calculate_reconstruction_statistics(self, results, input_pcs):
        """
        Calculate comprehensive reconstruction statistics
        """
        input_points = sum(len(pc) for pc in input_pcs)
        
        # Calculate surface area and volume
        vertices = results['final_vertices']
        triangles = results['final_triangles']
        
        surface_area = self.calculate_surface_area(vertices, triangles)
        
        # Pass method to volume calculation for appropriate threshold
        merged_points = results.get('merged_cloud', vertices)
        method = results.get('method', 'default')
        volume_data = results.get('volume_data', None)
        volume = self.calculate_volume(vertices, triangles, method = method, original_points = merged_points, voxel_grid_data = volume_data)
        
        # Validate results
        is_manifold = results['surface_quality']['is_manifold']
        is_closed, boundary_ratio = self.check_mesh_closure(triangles)
        
        if not is_manifold:
            self.log("Mesh is not manifold - some metrics may be inaccurate")
        
        if not is_closed:
            self.log(f"Mesh has {boundary_ratio*100:.1f}% open edges - volume is approximate")
        
        # Overall quality assessment
        merge_score = results['merge_quality']['coverage_balance']
        mesh_score = results['mesh_quality']['quality_score']
        surface_score = results['surface_quality']['quality_score']
        
        overall_quality = (merge_score + mesh_score + surface_score) / 3
        
        return {
            'input_points': input_points,
            'merged_points': len(results['merged_cloud']),
            'final_vertices': len(vertices),
            'final_triangles': len(triangles),
            'holes_filled': results['hole_info']['holes_filled'],
            'surface_area': surface_area,
            'volume': volume,
            'overall_quality': overall_quality,
            'method_used': 'First Principles Implementation',
            'is_manifold': is_manifold,
            'is_closed': is_closed,
            'boundary_edge_ratio': boundary_ratio,
            'volume_method': 'divergence_theorem' if is_closed else 'convex_hull_approximation'
        }
        
    def calculate_surface_area(self, vertices, triangles):
        """
        Calculate surface area by summing triangle areas
        """
        total_area = 0
        
        for triangle in triangles:
            vertex1, vertex2, vertex3 = vertices[triangle]
            
            # Calculate triangle area using cross product
            edge1 = vertex2 - vertex1
            edge2 = vertex3 - vertex1
            cross_product = np.cross(edge1, edge2)
            area = np.linalg.norm(cross_product) / 2
            
            total_area += area
            
        return total_area
    
    def calculate_volume(self, vertices, triangles, method='default', original_points=None, voxel_grid_data=None):
        """
        Calculate volume using appropriate method for reconstruction type
        """
        # Use method-appropriate threshold
        if method == 'grid_based':
            threshold = 0.20  # FIX: Was incorrectly 0.05
            
            # PREFERRED: Use voxel counting if grid data available
            if voxel_grid_data is not None:
                grid, voxel_size = voxel_grid_data
                n_occupied = np.sum(grid)
                volume = n_occupied * (voxel_size ** 3)
                self.log(f"Voxel counting: {n_occupied} voxels × {voxel_size}³ = {volume:.6f}m³")
                return volume
        else:
            threshold = 0.05
            
        # Check mesh closure
        is_closed, boundary_ratio = self.check_mesh_closure(triangles, threshold)
        
        if not is_closed:
            self.log(f"Mesh has {boundary_ratio*100:.1f}% open edges")
            # For voxel-based, approximate using original point cloud density
            if method == 'grid_based' and original_points is not None:
                return self.calculate_volume_voxel_approximation(original_points, voxel_size=0.007)
            # Last resort: bounding box (not convex hull - that overestimates)
            return self.calculate_volume_bounding_box(vertices)
            
        # Divergence theorem for closed meshes
        self.log(f"Closed mesh - using divergence theorem")
        total_volume = 0
        for triangle in triangles:
            v0, v1, v2 = vertices[triangle]
            signed_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            total_volume += signed_volume
        return abs(total_volume)    
        
    def calculate_volume_voxel_approximation(self, points, voxel_size=0.007):
        """
        Approximate volume by voxel counting on point cloud
        More accurate than convex hull for plants
        """
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int) + 2
        
        grid = np.zeros(grid_dims, dtype=bool)
        for point in points:
            idx = np.floor((point - min_coords) / voxel_size).astype(int) + 1
            idx = np.clip(idx, 0, grid_dims - 1)
            grid[tuple(idx)] = True
        
        volume = np.sum(grid) * (voxel_size ** 3)
        self.log(f"Voxel approximation: {np.sum(grid)} voxels = {volume:.6f}m³")
        return volume
    #def calculate_volume(self, vertices, triangles, method = 'default', original_points = None):
        """
        Calculate volume using divergence theorem ( if mesh is closed)
        else
        Use fallback methods like convex hull
        """ """
        # Use method-appropriate threshold
        if method == 'grid_based':
            threshold = 0.05  # 20% - relaxed for voxelized surfaces
        else:
            threshold = 0.05  # 5% - strict for smooth meshes
            
        # Check if mesh is closed or not
        is_closed, boundary_ratio = self.check_mesh_closure(triangles, threshold)
        
        if not is_closed:
            self.log(f"Mesh has {boundary_ratio*100:.1f}% open edges - using convex hull approximation")
            if original_points is not None and len(original_points) >= 4:
                self.log(f"  Using original point cloud ({len(original_points)} points) for volume")
                return self.calculate_volume_convex_hull(original_points)
            else:
                return self.calculate_volume_convex_hull(vertices)
            
        # Use accurate divergence theorem
        self.log(f"Mesh is closed ({boundary_ratio*100:.1f}% boundary edges) - using divergence theorem")
        total_volume = 0
        
        for triangle in triangles:
            vertex1, vertex2, vertex3 = vertices[triangle]
            
            # Calculate signed volume of tetrahedron formed by origin and triangle
            signed_volume = np.dot(vertex1, np.cross(vertex2, vertex3)) / 6.0
            total_volume += signed_volume
            
        return abs(total_volume)
        """
        
    def check_mesh_closure(self, triangles, threshold = 0.20):
        """
        Check if the mesh is closed using verification of edges if they shared them
        by exactlt 2 triangles
        - 0.05 (5%) = Strict, for CAD models/watertight meshes
        - 0.20 (20%) = Relaxed, for voxelized/grid-based meshes
        """
        edge_count = {}
        
        for triangle in triangles:
            edges = [tuple(sorted([triangle[0], triangle[1]])),
                     tuple(sorted([triangle[1], triangle[2]])),
                     tuple(sorted([triangle[2], triangle[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
                
        # Check if all edges are shared by exactly 2 triangles 
        boundary_edges = sum(1 for count in edge_count.values() if count == 1)
        
        # If less than 5% of edges are boundary eddges, consider it closed enough
        total_edges = len(edge_count)
        boundary_ratio = boundary_edges / total_edges if total_edges > 0 else 1.0
        
        return boundary_ratio < threshold, boundary_ratio
        
    def calculate_volume_convex_hull(self, vertices):
        """
        Fallback volume calculation method using hull approximation
        This approach provides a rough estimation for open meshes
        Uses simplified QuickHull algorithm
        """
        # Check if enough vertices are present
        if len(vertices) < 4:
            self.log("Warning: Not enough vertices for convex hull, using bounding box")
            return self.calculate_volume_bounding_box(vertices)
            
        # Compute convex hull
        hull_vertices, hull_faces = self.convex_hull_3d(vertices)
        
        # Check if computation is successful or not
        if len(hull_faces) == 0:
            self.log("Warning: Convex hull computation produced no faces, using bounding box")
            return self.calculate_hull_volume_bounding_box(vertices)
            
        # Calculate volume from hull
        volume = self.calculate_hull_volume(hull_vertices, hull_faces)
        
        # Validate volume if reasonable
        if volume <= 0 or np.isnan(volume) or np.isinf(volume):
            self.log("Warning: Invalid convex hull volume, using bounding box")
            return self.calculate_volume_bounding_box(vertices)
            
        self.log(f"Convex hull volume: {volume:.6f}m3")
        
        return volume
        
    def convex_hull_3d(self, points):
        """
        Simplified 3D convex hull using gift wrapping for plants
        """
        if len(points) < 4:
            return points, np.array([])
            
        # Remove dups
        points = np.unique(points, axis=0)
        n = len(points)
        
        if n < 4:
            return points, np.array([])
            
        # Find the extreme points in each direction to form the initial huill
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        min_y_idx = np.argmin(points[:, 1])
        max_y_idx = np.argmax(points[:, 1])
        min_z_idx = np.argmin(points[:, 2])
        max_z_idx = np.argmax(points[:, 2])
        
        extreme_indices = np.unique([min_x_idx, max_x_idx, min_y_idx, max_y_idx, min_z_idx, max_z_idx])
        
        # Uses extrematies to build the initial hull
        hull_faces = []
        
        # Create faces between extreme points
        for i in range(len(extreme_indices)):
            for j in range(i + 1, len(extreme_indices)):
                for k in range(j + 1, len(extreme_indices)):
                    index1, index2, index3 = extreme_indices[i], extreme_indices[j], extreme_indices[k]
                    
                    # Check if face has all other points on one side
                    face_points = points[[index1, index2, index3]]
                    normal = self.compute_triangle_normal(face_points)
                    face_center = np.mean(face_points, axis = 0)
                    
                    # Check all points
                    all_on_side = True
                    for point_index in range(n):
                        if point_index in [index1, index2, index3]:
                            continue
                            
                        to_point = points[point_index] - face_center
                        if np.dot(to_point, normal) > 1e-6:
                            all_on_side = False
                            break
                            
                    if all_on_side:
                        hull_faces.append([index1, index2, index3])
                        
        if len(hull_faces) == 0:
            # Use the extreme points for a simple triangulation approach
            self.log("Using simplified hull construction")
            hull_faces = self.simple_hull(points, extreme_indices)
            
        # Extract unique hull vertices and remap face indices
        hull_faces_array = np.array(hull_faces, dtype = int)
        unique_vertex_indices = np.unique(hull_faces_array.flatten())
        hull_vertices = points[unique_vertex_indices]
        
        # Create mapping from old indices to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
        
        # Remap
        remapped_faces = []
        for face in hull_faces:
            remapped_face = [index_map[index] for index in face]
            remapped_faces.append(remapped_face)
            
        return hull_vertices, np.array(remapped_faces, dtype = int)
        
    def simple_hull(self, points, extreme_indices):
        """
        Create simple hull using triangulation on extreme points
        """
        faces = []
    
        # Create all possible triangular faces from extreme points
        n_extremes = len(extreme_indices)
        
        for i in range(n_extremes):
            for j in range(i + 1, n_extremes):
                for k in range(j + 1, n_extremes):
                    idx1 = extreme_indices[i]
                    idx2 = extreme_indices[j]
                    idx3 = extreme_indices[k]
                    faces.append([idx1, idx2, idx3])
        
        # If no faces created or we have exactly 4 points, create tetrahedron
        if len(faces) == 0 and len(extreme_indices) >= 4:
            faces = [
                [extreme_indices[0], extreme_indices[1], extreme_indices[2]],
                [extreme_indices[0], extreme_indices[1], extreme_indices[3]],
                [extreme_indices[0], extreme_indices[2], extreme_indices[3]],
                [extreme_indices[1], extreme_indices[2], extreme_indices[3]]
            ]
        
        return faces
        
    def compute_triangle_normal(self, triangle_points):
        """
        Compute the normal vector for a triangle
        """
        if len(triangle_points) != 3:
            return np.array([0, 0, 1])
        
        vertex1, vertex2, vertex3 = triangle_points
        edge1 = vertex2 - vertex1
        edge2 = vertex3 - vertex1
        normal = np.cross(edge1, edge2)
        
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-10:
            normal = normal / norm_length
        
        return normal
        
    def calculate_hull_volume(self, vertices, faces):
        """
        Calculate volume from convex hull faces
        """
        if len(faces) == 0:
            return 0.0
        
        total_volume = 0.0
        
        for face in faces:
            if len(face) != 3:
                continue
            
            v1, v2, v3 = vertices[face]
            
            # Signed volume of tetrahedron from origin to face
            signed_volume = np.dot(v1, np.cross(v2, v3)) / 6.0
            total_volume += signed_volume
        
        return abs(total_volume)
        
    def calculate_volume_bounding_box(self, vertices):
        """
        Estimation of volume using the bounding box for the plants. 
        Use about 20-30% of bounding box as heuritic
        """
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        
        # Bounding box volume
        bbox_volume = np.prod(dimensions)
        
        # Assume the usage of about 25% of the bounding box is used
        estimated_volume = bbox_volume * 0.25
        
        self.log(f"Bounding box: {dimensions[0]:.3f} × {dimensions[1]:.3f} × {dimensions[2]:.3f} m")
        self.log(f"Estimated volume (25% of bbox): {estimated_volume:.6f} m³")
        
        return estimated_volume
        
        
        
