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
        self.fx, self.fy = 365.60, 365.15
        self.cx, self.cy = 248.82, 208.63
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
        
        self.report_progress("Pipeline complete!", 100) 
        self.plant_count += 1   
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
