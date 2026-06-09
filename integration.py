""" Imports for all my processing classes """
from classes.preprocessing_class import PreProcessing
from classes.registration_class import Registration
from classes.reconstruction import ThreeDReconstruction

""" Imports for all Prediction classes """
from classes.ann_class import BiomassANN
from classes.random_forest_class import BiomassRandomForest, RandomForestRegressor, DecisionTreeRegressor

""" Standard library imports """
import numpy as np
try:
    import open3d as o3d
except Exception:
    o3d = None
import matplotlib.pyplot as plt
import socket
import threading
import serial
import subprocess
import time
import sys
import os

class Integration:
    def __init__(self, progress_callback = None):
        self.running = True
        self.plant_count = 1
        
        # Processing variables initialization
        # Parameters
        self.icp_param = {'max_iterations': 300, 'tolerance': 1e-6, 'max_corr_dist': 0.10}

        # Intrinsic parameters for Microsoft Kinect v2 IR/depth stream @ 512x424.
        # These are the published factory defaults for the Kinect v2 IR camera.
        # For best accuracy they can be overridden at runtime from the device:
        #   params = device.getIrCameraParams()
        #   fx, fy, cx, cy = params.fx, params.fy, params.cx, params.cy
        self.fx, self.fy = 365.456, 365.456
        self.cx, self.cy = 254.878, 205.395
        self.x_min, self.x_max = -0.5, 0.5
        self.y_min, self.y_max = -0.6, 0.65
        self.z_min, self.z_max = 0.2, 1.5
        self.RADIUS = 0.13

        # Initialize classes
        self.preprocessor = PreProcessing()
        self.registration = Registration()
        self.reconstruction = ThreeDReconstruction(verbose=True)
        
        # ANN and RF Biomass Prediction
        self.prediction_ann = None
        self.prediction_rf = None
        
        # Progress callback 
        self.progress_callback = progress_callback 
    
    def report_progress(self, message, percentage = None):
        """
        Report progress to callback if available
        """
        if self.progress_callback:
            self.progress_callback(message, percentage)
        
    def start(self, count):
        # Start timing
        start_time = time.time()
        self.plant_count = count
        
        self.report_progress("Loading depth map...", 5)
        
        # Processing code goes here onwards
        filename = 'data_collection'
        
        # Loading the data
        _0_depth_map = np.load(f'{filename}/0_degrees_depth_plant_{self.plant_count}.npy')
        _90_depth_map = np.load(f'{filename}/90_degrees_depth_plant_{self.plant_count}.npy')
        _180_depth_map = np.load(f'{filename}/180_degrees_depth_plant_{self.plant_count}.npy')
        _270_depth_map = np.load(f'{filename}/270_degrees_depth_plant_{self.plant_count}.npy')
        
        print("step 1: Load done")
        self.report_progress("Preprocessing point clouds...", 15)
        
        # Preprocessing pipeline
        _0_points, _0_surface_normals, _0_pixels = self.preprocessor.complete_preprocessing_pipeline(
        _0_depth_map, self.fx, self.fy, self.cx, self.cy, self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        _90_points, _90_surface_normals, _90_pixels = self.preprocessor.complete_preprocessing_pipeline(_90_depth_map, self.fx, self.fy, self.cx, self.cy, self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        _180_points, _180_surface_normals, _180_pixels = self.preprocessor.complete_preprocessing_pipeline(_180_depth_map, self.fx, self.fy, self.cx, self.cy, self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        _270_points, _270_surface_normals, _270_pixels = self.preprocessor.complete_preprocessing_pipeline(_270_depth_map, self.fx, self.fy, self.cx, self.cy, self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        point_clouds = [_0_points, _90_points, _180_points, _270_points]
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        
        print("step 2: preprocess done")
        self.report_progress("Registering point clouds...", 40)
        # Registration
        # Coarse Registration
        arranged_model, construction_center = self.registration.arrange_views_in_circle(point_clouds, angles, self.RADIUS)
        
        # Fine Registration
        fine_registered_pcs, transformations, registration_stats = self.registration.sequential_icp_registration(arranged_model, self.icp_param)
        print("step 3: registration done")
        self.report_progress("Reconstructing 3D mesh...", 60)
        # Reconstruction Params
        # Option 1: MULTI-VIEW PROJECTION (RECOMMENDED - Very Fast!)
        #mesh_method = 'projection'
        #reconstruction_params = {
        #    'grid_resolution': 0.005,  # 5mm grid (adjust for detail vs speed)
        #    'hole_threshold': 0.10,
        #    'smooth_iterations': 20,
        #    'preserve_features': True
        #}
        
        # Option 2: GREEDY PROJECTION TRIANGULATION (Fast)
        #mesh_method = 'greedy_projection'
        #reconstruction_params = {
        #    'max_edge_length': 0.03,  # Maximum edge length in meters
        #    'hole_threshold': 0.10,
        #    'smooth_iterations': 20,
        #    'preserve_features': True
        #}
        
        # Option 3: GRID-BASED SURFACE MESH (Very Fast!)
        mesh_method = 'grid_based'
        reconstruction_params = {
            'voxel_size': 0.007,  
            'hole_threshold': 0,
            'smooth_iterations': 0,
            'preserve_features': True,
            'fill_holes': False
        }
        
        # Run reconstruction pipeline
        reconstruction_results = self.reconstruction.complete_reconstruction_pipeline(
            fine_registered_pcs,
            method=mesh_method,
            **reconstruction_params
        )
        
        # RESULTS ANALYSIS
        stats = reconstruction_results['reconstruction_stats']
        merge_qual = reconstruction_results['merge_quality']
        mesh_qual = reconstruction_results['mesh_quality']
        surface_qual = reconstruction_results['surface_quality']
        hole_info = reconstruction_results['hole_info']
    
        # Calculate the dimensions of the plant
        final_vertices = reconstruction_results['final_vertices']
        final_triangles = reconstruction_results['final_triangles']

        # Calculate bounding box dimensions from final mesh vertices
        x_coords = final_vertices[:, 0]
        y_coords = final_vertices[:, 1]
        z_coords = final_vertices[:, 2]

        width = (x_coords.max() - x_coords.min()) * 100  # Convert to cm
        height = (y_coords.max() - y_coords.min()) * 100
        depth = (z_coords.max() - z_coords.min()) * 100

        # Save Results
        print("Saving results")
        self.report_progress("Saving reconstruction files...", 86)
        output_dir = "reconstruction_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}/")

        # Save point cloud as PLY using Open3D
        if o3d is not None:
            pcd_final = o3d.geometry.PointCloud()
            pcd_final.points = o3d.utility.Vector3dVector(reconstruction_results['merged_cloud'])

            # Color by height using viridis gradient (instead of by view)
            merged_cloud = reconstruction_results['merged_cloud']
            y_coords = merged_cloud[:, 1]
            if y_coords.max() != y_coords.min():
                y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
            else:
                y_normalized = y_coords * 0.0
            point_colors = plt.cm.viridis(y_normalized)[:, :3]  # Use viridis gradient
            pcd_final.colors = o3d.utility.Vector3dVector(point_colors)

            # Save with the plant number in filename
            try:
                o3d.io.write_point_cloud(f"{output_dir}/merged_point_cloud_plant_{self.plant_count}.ply", pcd_final)
            except Exception as e:
                print(f"Warning: failed to write point cloud with Open3D: {e}")

            # Save mesh as PLY and OBJ
            try:
                mesh_final = o3d.geometry.TriangleMesh()
                mesh_final.vertices = o3d.utility.Vector3dVector(reconstruction_results['final_vertices'])
                mesh_final.triangles = o3d.utility.Vector3iVector(reconstruction_results['final_triangles'])
                mesh_final.compute_vertex_normals()

                o3d.io.write_triangle_mesh(f"{output_dir}/final_mesh_plant_{self.plant_count}.ply", mesh_final)
                o3d.io.write_triangle_mesh(f"{output_dir}/final_mesh_plant_{self.plant_count}.obj", mesh_final)
            except Exception as e:
                print(f"Warning: failed to write mesh with Open3D: {e}")
        else:
            print("Open3D not available — skipping PLY/OBJ export. Numpy arrays will still be saved.")

        # Save numpy arrays
        np.save(f"{output_dir}/merged_points_plant_{self.plant_count}.npy", reconstruction_results['merged_cloud'])

        np.save(f"{output_dir}/final_vertices_plant_{self.plant_count}.npy", reconstruction_results['final_vertices'])

        np.save(f"{output_dir}/final_triangles_plant_{self.plant_count}.npy", reconstruction_results['final_triangles'])

        np.save(f"{output_dir}/surface_normals_plant_{self.plant_count}.npy", reconstruction_results['surface_normals'])
        
        # Save statistics
        with open(f"{output_dir}/reconstruction_stats_plant_{self.plant_count}.txt", 'w') as f:
            f.write("3D RECONSTRUCTION RESULTS (FU4.4)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("INPUT DATA:\n")
            f.write(f"  Views processed: 4 (0Â°, 90Â°, 180Â°, 270Â°)\n")
            f.write(f"  Total input points: {stats['input_points']:,}\n\n")
            
            f.write("MERGING RESULTS:\n")
            f.write(f"  Merged points: {stats['merged_points']:,}\n")
            f.write(f"  Coverage balance: {merge_qual['coverage_balance']:.3f}\n")
            f.write(f"  Density uniformity: {merge_qual['density_uniformity']:.3f}\n\n")
            
            f.write("FINAL MESH:\n")
            f.write(f"  Vertices: {stats['final_vertices']:,}\n")
            f.write(f"  Triangles: {stats['final_triangles']:,}\n")
            f.write(f"  Surface area: {stats['surface_area']:.6f} mÂ²\n")
            f.write(f"  Volume: {stats['volume']:.8f} mÂ³\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write(f"  Overall quality: {stats['overall_quality']:.3f}\n")
            f.write(f"  Geometric fidelity: {surface_qual['geometric_fidelity']:.3f}\n")
            f.write(f"  Surface smoothness: {surface_qual['smoothness']:.3f}\n")
            f.write(f"  Is manifold: {surface_qual['is_manifold']}\n")
            f.write(f"  Holes filled: {hole_info['holes_filled']}\n\n")
            
            f.write("PLANT DIMENSIONS:\n")
            f.write(f"  Height (Y): {height:.2f} cm\n")
            f.write(f"  Width (X): {width:.2f} cm\n")
            f.write(f"  Depth (Z): {depth:.2f} cm\n\n")
        
        """ Get the Biomass using ANN 
        print("Calculating Biomass using ANN")
        # Configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(script_dir, "ANN_model", "biomass_ann_model")
        RECONSTRUCTION_DIR = os.path.join(script_dir, "reconstruction_output")
        WEIGHTS_FILE = os.path.join(script_dir, "weights.txt")
        
        # Features used during training (MUST MATCH TRAINING)
        SELECTED_FEATURES = [
            'volume',
            'surface_area', 
            'height',
            'compactness',
            'overall_quality'
        ]
        
        model = BiomassANN()
        model.load_model(MODEL_PATH)
        """
        
        """ Get the Biomass using RF """
        print("Calculating Biomass using RF")
        self.report_progress("Predicting biomass...", 95)
        # Configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(script_dir, "RF_model", "biomass_rf_model")
        RECONSTRUCTION_DIR = os.path.join(script_dir, "reconstruction_output")
        WEIGHTS_FILE = os.path.join(script_dir, "weights.txt")
        
        # Features used during training (MUST MATCH TRAINING)
        SELECTED_FEATURES = [
            'volume',
            'surface_area', 
            'height',
            'bbox_volume',
            'surface_to_volume_ratio',
            'height_to_volume_ratio'
        ]
        
        # Load training model
        model2 = BiomassRandomForest()
        model2.load_model(MODEL_PATH)
        
         # Extract features
        features_dict = model2.extract_features_from_reconstruction(RECONSTRUCTION_DIR, self.plant_count)
        
        # Convert to array in correct order
        X = np.array([[features_dict[feat] for feat in SELECTED_FEATURES]])
        
        # Predict using your existing predict() method
        self.prediction_rf = model2.predict(X)[0]
        
        # Calculate the execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save Biomass Stats
        with open(f"{output_dir}/reconstruction_stats_plant_{self.plant_count}.txt", 'a') as f:
            f.write("PLANT BIOMASS ESTIMATION:\n")
            f.write(f"  RF Biomass Prediction: {self.prediction_rf:.2f}kg\n\n")
            
            f.write("PROCESSING TIME: \n")
            f.write(f"  Total execution time: {execution_time:.2f} seconds\n")
        
        self.report_progress("Generating visualisation...", 98)
        self.generate_reconstruction_png(
            reconstruction_results=reconstruction_results,
            stats=stats,
            merge_qual=merge_qual,
            surface_qual=surface_qual,
            features_dict=features_dict,
            biomass_rf=self.prediction_rf,
            width=width, height=height, depth=depth,
            output_dir=output_dir
        )

        self.report_progress("Pipeline complete!", 100)
        self.plant_count += 1

    # =================================================================================
    # Visualisation
    # =================================================================================

    def generate_reconstruction_png(self, reconstruction_results, stats, merge_qual,
                                     surface_qual, features_dict, biomass_rf,
                                     width, height, depth, output_dir):
        """
        Produce a multi-panel PNG that visualises the reconstructed plant and shows
        how the 3D morphological features relate to the predicted biomass.

        Layout (2 rows × 3 cols):
          Row 1:  Front projection | Side projection | Top projection
          Row 2:  3-D mesh         | Plant stats     | Feature → Biomass bar chart
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        merged_cloud   = reconstruction_results['merged_cloud']
        final_vertices = reconstruction_results['final_vertices']
        final_triangles = reconstruction_results['final_triangles']

        # ── Colour point cloud by height (Y axis) ──────────────────────────────
        y = merged_cloud[:, 1]
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        colours = plt.cm.viridis(y_norm)

        # Subsample for scatter plots (keep render fast)
        n_pts   = len(merged_cloud)
        stride  = max(1, n_pts // 4000)
        idx     = np.arange(0, n_pts, stride)
        pts_sub = merged_cloud[idx]
        col_sub = colours[idx]

        fig = plt.figure(figsize=(18, 11), facecolor='#1a1a2e')
        fig.suptitle(
            f'3-D Reconstruction  |  Plant {self.plant_count - 1}  |  '
            f'Predicted Biomass: {biomass_rf:.2f} kg',
            fontsize=16, color='white', fontweight='bold', y=0.98
        )

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.38, wspace=0.32,
                               left=0.05, right=0.97,
                               top=0.93, bottom=0.06)

        ax_style = dict(facecolor='#0d0d1a')

        # ── Helper: style a 2-D projection axis ───────────────────────────────
        def style_ax(ax, xlabel, ylabel, title):
            ax.set_facecolor('#0d0d1a')
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            ax.set_xlabel(xlabel, color='#aaaaaa', fontsize=8)
            ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
            ax.set_title(title, color='white', fontsize=9, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        # ── Row 1: three orthographic projections ─────────────────────────────
        # Front  (X–Y)
        ax_front = fig.add_subplot(gs[0, 0], **ax_style)
        ax_front.scatter(pts_sub[:, 0], pts_sub[:, 1],
                         c=col_sub, s=0.8, linewidths=0)
        style_ax(ax_front, 'X (m)', 'Y – height (m)', 'Front view  (X – Y)')

        # Side  (Z–Y)
        ax_side = fig.add_subplot(gs[0, 1], **ax_style)
        ax_side.scatter(pts_sub[:, 2], pts_sub[:, 1],
                        c=col_sub, s=0.8, linewidths=0)
        style_ax(ax_side, 'Z (m)', 'Y – height (m)', 'Side view  (Z – Y)')

        # Top  (X–Z)
        ax_top = fig.add_subplot(gs[0, 2], **ax_style)
        ax_top.scatter(pts_sub[:, 0], pts_sub[:, 2],
                       c=col_sub, s=0.8, linewidths=0)
        style_ax(ax_top, 'X (m)', 'Z (m)', 'Top view  (X – Z)')

        # ── Row 2, Col 0: 3-D mesh ────────────────────────────────────────────
        ax_mesh = fig.add_subplot(gs[1, 0], projection='3d',
                                  facecolor='#0d0d1a')
        ax_mesh.set_facecolor('#0d0d1a')

        # Subsample triangles so rendering stays fast
        tri_stride = max(1, len(final_triangles) // 3000)
        tri_sub    = final_triangles[::tri_stride]

        # Colour faces by mean Y height of their vertices
        face_y = final_vertices[tri_sub, 1].mean(axis=1)
        face_y_norm = (face_y - face_y.min()) / (face_y.max() - face_y.min() + 1e-9)
        face_colours = plt.cm.viridis(face_y_norm)

        polys = [final_vertices[t] for t in tri_sub]
        mesh_coll = Poly3DCollection(polys, alpha=0.55,
                                     facecolors=face_colours,
                                     edgecolors='none')
        ax_mesh.add_collection3d(mesh_coll)

        # Set axis limits from point cloud bounding box
        for axis, dim in zip(['x', 'y', 'z'], [0, 1, 2]):
            lo, hi = final_vertices[:, dim].min(), final_vertices[:, dim].max()
            getattr(ax_mesh, f'set_{axis}lim')(lo, hi)

        ax_mesh.set_xlabel('X', color='#aaaaaa', fontsize=7)
        ax_mesh.set_ylabel('Z', color='#aaaaaa', fontsize=7)
        ax_mesh.set_zlabel('Y', color='#aaaaaa', fontsize=7)
        ax_mesh.set_title('3-D Mesh  (height-coloured)', color='white', fontsize=9)
        ax_mesh.tick_params(colors='#aaaaaa', labelsize=6)
        ax_mesh.xaxis.pane.fill = False
        ax_mesh.yaxis.pane.fill = False
        ax_mesh.zaxis.pane.fill = False

        # ── Row 2, Col 1: plant stats text panel ──────────────────────────────
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.set_facecolor('#0d0d1a')
        ax_stats.axis('off')

        vol_L   = stats['volume'] * 1000          # m³ → litres
        area_dm = stats['surface_area'] * 100     # m² → dm²

        lines = [
            ('Plant dimensions', ''),
            ('  Height',  f"{height:.1f} cm"),
            ('  Width',   f"{width:.1f} cm"),
            ('  Depth',   f"{depth:.1f} cm"),
            ('', ''),
            ('Mesh quality', ''),
            ('  Vertices',      f"{stats['final_vertices']:,}"),
            ('  Triangles',     f"{stats['final_triangles']:,}"),
            ('  Surface area',  f"{area_dm:.1f} dm²"),
            ('  Volume',        f"{vol_L:.3f} L"),
            ('  Quality score', f"{stats['overall_quality']:.3f}"),
            ('  Geo. fidelity', f"{surface_qual['geometric_fidelity']:.3f}"),
            ('  Is manifold',   str(surface_qual['is_manifold'])),
            ('', ''),
            ('Biomass (RF)',  f"{biomass_rf:.2f} kg"),
        ]

        y_pos = 0.97
        for label, value in lines:
            if label == '' and value == '':
                y_pos -= 0.04
                continue
            if value == '':          # section header
                ax_stats.text(0.03, y_pos, label, color='#7ec8e3',
                              fontsize=8.5, fontweight='bold',
                              transform=ax_stats.transAxes, va='top')
            else:
                ax_stats.text(0.03, y_pos, label, color='#cccccc',
                              fontsize=8, transform=ax_stats.transAxes, va='top')
                ax_stats.text(0.62, y_pos, value, color='white',
                              fontsize=8, fontweight='bold',
                              transform=ax_stats.transAxes, va='top')
            y_pos -= 0.058

        ax_stats.set_title('Plant Stats', color='white', fontsize=9, pad=4)

        # ── Row 2, Col 2: feature → biomass bar chart ─────────────────────────
        ax_bar = fig.add_subplot(gs[1, 2])
        ax_bar.set_facecolor('#0d0d1a')

        feat_labels = {
            'volume':                 'Volume (m³)',
            'surface_area':           'Surface area (m²)',
            'height':                 'Height (m)',
            'bbox_volume':            'Bbox volume (m³)',
            'surface_to_volume_ratio':'SA : V ratio',
            'height_to_volume_ratio': 'H : V ratio',
        }

        bar_names  = []
        bar_values = []
        for key, label in feat_labels.items():
            if key in features_dict:
                bar_names.append(label)
                bar_values.append(features_dict[key])

        # Normalise values to [0, 1] for a visual comparison
        bar_arr  = np.array(bar_values, dtype=float)
        bar_norm = bar_arr / (bar_arr.max() + 1e-9)

        colours_bar = plt.cm.plasma(np.linspace(0.2, 0.85, len(bar_names)))
        bars = ax_bar.barh(bar_names, bar_norm, color=colours_bar,
                           edgecolor='none', height=0.6)

        # Annotate with raw values
        for bar, raw in zip(bars, bar_values):
            ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{raw:.3g}', va='center', color='white', fontsize=7)

        # Biomass marker line (normalised relative to max feature value)
        biomass_norm = biomass_rf / (bar_arr.max() + 1e-9)
        ax_bar.axvline(biomass_norm, color='#ff6b6b', linewidth=1.5,
                       linestyle='--', label=f'Biomass = {biomass_rf:.2f} kg')
        ax_bar.legend(loc='lower right', fontsize=7, framealpha=0.3,
                      labelcolor='white', facecolor='#0d0d1a')

        ax_bar.set_xlim(0, 1.25)
        ax_bar.set_title('Morphological Features → Biomass', color='white', fontsize=9, pad=4)
        ax_bar.tick_params(colors='#aaaaaa', labelsize=7)
        ax_bar.set_xlabel('Normalised value', color='#aaaaaa', fontsize=8)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor('#333355')

        # ── Colourbar (shared height gradient) ────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(
                                       vmin=merged_cloud[:, 1].min(),
                                       vmax=merged_cloud[:, 1].max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_front, ax_side, ax_top],
                            orientation='horizontal', fraction=0.025, pad=0.12,
                            aspect=40)
        cbar.set_label('Height (m)', color='#aaaaaa', fontsize=8)
        cbar.ax.tick_params(colors='#aaaaaa', labelsize=7)

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = f"{output_dir}/reconstruction_plant_{self.plant_count - 1}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved reconstruction PNG → {out_path}")

    # =================================================================================
    # Start helper functions
    # =================================================================================
    
    def capture(self, filenames, count):
        """
        Function which calls camera_red and camera_green to capture 
        from both devices
        """
        camera_red = self.capture_red(filenames[1], count)
        if camera_red:
            camera_green = self.capture_green(filenames[0], count)
            if camera_green:
                print("Both camera captures completed sucessfuly")
                return True
            else:
                print("Camera red was successful but camera green failed")
                print("Deleting camera red data")
        else:
            print("Camera capture failed! Please run command again")
            
        return False
            
                    
    def capture_red(self, filename, count):
        """
        Function to capture from just the red camera using camera_red.py
        """
        # Runs the camera capture code as a subroutine
        print(f"Running camera capture camera_red.py...")
        result = subprocess.run([sys.executable, "classes/camera_red.py", filename, str(count)], capture_output = True, text = True)
        
        if result.returncode == 0:
            print("Camera capture completed successfully")
            print("Capture output:", result.stdout)
            return True
        else:
            print("Camera capture failed")
            print("Error:", result.stderr)
            return False
        
    def capture_green(self, filename, count):
        """
        Function to capture from just the green camera using camera_green.py
        """
        # Runs the camera capture code as a subroutine
        print(f"Running camera capture camera_green.py...")
        result = subprocess.run([sys.executable, "classes/camera_green.py", filename, str(count)], capture_output = True, text = True)
        
        if result.returncode == 0:
            print("Camera capture completed successfully")
            print("Capture output:", result.stdout)
            return True
        else:
            print("Camera capture failed")
            print("Error:", result.stderr)
            return False
            
    # ==================================================================================
    # Arduino Connection code
    # ==================================================================================
        
    def setup_arduino(self):
        # Setup arduino serial comms
        if self.connect_arduino():
            print(f"Arduino connected on {self.arduino_port}")
        else:
            print(f"Failed to connect to Arduino on {self.arduino_port}")
            print(f"Serial commands will be disabled")
            
    def connect_arduino(self):
        # Connect to arduino via serial comms
        success = True
        error_occurred = False
        
        # Set timout for connection attempt
        connection_timeout = 5
        start_time = time.time()
        
        # Check if port is available and exists
        if not os.path.exists(self.arduino_port):
            error_occurred = True
            success = False
            print(f"Arduino port {self.arduino_port} does not exist")
        else:
            # attempt serial connection
            self.arduino = serial.Serial()
            self.arduino.port = self.arduino_port
            self.arduino.baudrate = self.baud_rate
            self.arduino.timeout = 1
            
            # Check if we can open connection
            if hasattr(self.arduino, 'open'):
                self.arduino.open()
                if self.arduino.is_open:
                    time.sleep(2)
                    success = True
                else:
                    error_occurred = True
                    success = False
            else:
                error_occurred = True
                success = False
                
        if error_occurred:
            self.arduino = None
            
        return success
        
    def send_arduino_command(self, command):
        # Send the command to the board
        if self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
            send_success = True
            error_occurred = False
            
            # Check if still connected
            if hasattr(self.arduino, 'write'):
                # Attempt to write
                bytes_written = 0
                command_bytes = command.encode()
                
                # check if write operation is available
                if hasattr(self.arduino, 'write'):
                    write_result = self.write_to_arduino(command_bytes)
                    if write_result:
                        time.sleep(0.1)
                        send_success = True
                    else:
                        error_occurred = True
                        send_success = False
                else:
                    error_occurred = True
                    send_success = False
            else:
                error_occurred = True
                send_success = False
                
            if error_occurred:
                print("Failed to send command to Arduino")
                return False
            else:
                print(f"Sent '{command}' to Arduino")
                return True
        else:
            print("Arduino not connected")
            return False
            
    def write_to_arduino(self, data):
        # Write data to arduino
        success = True
        
        if self.arduino and hasattr(self.arduino, 'write') and self.arduino.is_open:
            # Perform write operation
            result = self.arduino.write(data)
            if result > 0:
                success = True
            else:
                success = False
        else:
            success = False
            
        return success
