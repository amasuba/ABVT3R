"""
Host server running on the NVIDIA Jetson Nano.
Manages bidirectional TCP socket communications with the GUI client,
controls the Arduino stepper-motor gantry, triggers Intel RealSense
camera captures, and executes the full processing pipeline.
"""
# import the Integration class
from integration import Integration
from classes.random_forest_class import RandomForestRegressor, DecisionTreeRegressor
# Standard imports 
import socket
import threading
import time
import subprocess
import sys
import os
import serial
import serial.tools.list_ports
import glob
import struct

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available")
    
class Host:
    def __init__(self, port = 8888, arduino_port = '/dev/ttyACM0', baud_rate = 9600, progress_callback = None):
        self.port = port
        self.client_socket = None
        self.running = True
        self.server_socket = None
        self.client_connected = False
        
        # Progress callback
        self.progress_callback = progress_callback
        
        # Arduino serial connection init
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.arduino = None
        self.setup_arduino()
        
        # Plant count tracking
        self.plant_count = 1
        
        # Integration class instance
        self.integration = None
    
    def report_progress(self, message, percentage = None):
        """
        Report progress to GUI
        """        
        # Send progress to client over socket
        if self.client_socket and self.client_connected:
            progress_msg = f"PROGRESS:{percentage}:{message}"
            self.client_socket.send(progress_msg.encode('utf-8'))
            time.sleep(0.05)  # Small delay to avoid overwhelming the socket
    # =============================================================================
    # File Transfer Functions
    # =============================================================================
    
    def send_file(self, filepath):
        """
        Send a file to the client over the socket
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        # Get filename and file size
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)
        
        print(f"Sending file: {filename} ({filesize} bytes)")
        
        # Send filename length (4 bytes)
        filename_bytes = filename.encode('utf-8')
        filename_length = len(filename_bytes)
        self.client_socket.sendall(struct.pack('!I', filename_length))
        
        # Send filename
        self.client_socket.sendall(filename_bytes)
        
        # Send file size (8 bytes for large files)
        self.client_socket.sendall(struct.pack('!Q', filesize))
        
        # Send file data in chunks
        with open(filepath, 'rb') as f:
            sent = 0
            while sent < filesize:
                chunk = f.read(4096)  # 4KB chunks
                if not chunk:
                    break
                self.client_socket.sendall(chunk)
                sent += len(chunk)
                
                # Progress indicator
                progress = (sent / filesize) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        print(f"\nFile sent successfully: {filename}")
        
        return True
        
    def send_all_plant_files(self, plant_number):
        """
        Send all files related to the specified plant number
        """
        print(f"\n{'='*70}")
        print(f"TRANSFERRING FILES FOR PLANT {plant_number}")
        print(f"{'='*70}\n")
        
        output_dir = "reconstruction_output"
        
        # List of files to transfer
        files_to_send = [
            f"merged_point_cloud_plant_{plant_number}.ply",
            f"final_mesh_plant_{plant_number}.ply",
            f"merged_points_plant_{plant_number}.npy",
            f"final_vertices_plant_{plant_number}.npy",
            f"final_triangles_plant_{plant_number}.npy",
            f"surface_normals_plant_{plant_number}.npy",
            f"reconstruction_stats_plant_{plant_number}.txt"
        ]
        
        # Send START_TRANSFER signal
        self.send_message("START_TRANSFER")
        time.sleep(1)
        
        # Send number of files
        files_exist = []
        for filename in files_to_send:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                files_exist.append(filepath)
        
        num_files = len(files_exist)
        self.send_message(f"NUM_FILES:{num_files}")
        time.sleep(1)
        
        # Send each file
        success_count = 0
        for filepath in files_exist:
            if self.send_file(filepath):
                success_count += 1
                time.sleep(1)  # Small delay between files
        
        print(f"\n{'='*70}")
        print(f"FILE TRANSFER COMPLETE: {success_count}/{num_files} files sent")
        print(f"{'='*70}\n")
        
        return success_count == num_files
    
    # ==================================================================================
    # Arduino Connection Code (from integration.py)
    # ==================================================================================
    
    def find_arduino_port(self):
        """
        Scan all /dev/ttyACM* ports to find Arduino
        Returns the port path if found, None otherwise
        """
        print("Scanning for Arduino on all ttyACM ports...")
        
        # Get all ACM ports
        acm_ports = glob.glob('/dev/ttyACM*')
        
        if not acm_ports:
            print("No /dev/ttyACM* ports found")
            return None
        
        # Use serial.tools.list_ports to get detailed info
        available_ports = list(serial.tools.list_ports.comports())
        
        for acm_port in acm_ports:
            print(f"Checking {acm_port}...")
            
            # Find matching port info
            for port_info in available_ports:
                if port_info.device == acm_port:
                    # Check if it's an Arduino
                    if 'Arduino' in str(port_info.description) or \
                       'Arduino' in str(port_info.manufacturer) or \
                       'USB' in str(port_info.description):
                        print(f"Found Arduino on {acm_port}")
                        print(f"  Description: {port_info.description}")
                        print(f"  Manufacturer: {port_info.manufacturer}")
                        return acm_port
        
        print("No Arduino found on any ttyACM port")
        return None
        
    def setup_arduino(self):
        """Setup arduino serial comms"""
        # Find the available ports
        found_port = self.find_arduino_port()
        
        if found_port:
            self.arduino_port = found_port
            if self.connect_arduino():
                print(f"Arduino connected on {self.arduino_port}")
                return True
                
        print(f"Failed to connect to Arduino on {self.arduino_port}")
        print(f"Serial commands will be disabled")
        
        return False
    
    def connect_arduino(self):
        """Connect to arduino via serial comms"""
        success = True
        error_occurred = False
        
        # Close existing connection if any
        if self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
            try:
                self.arduino.close()
            except:
                pass
        
        # Check if port is available and exists
        if not os.path.exists(self.arduino_port):
            error_occurred = True
            success = False
            print(f"Arduino port {self.arduino_port} does not exist")
        else:
            try:
                # Attempt serial connection
                self.arduino = serial.Serial(
                    port=self.arduino_port,
                    baudrate=self.baud_rate,
                    timeout=1
                )
                
                if self.arduino.is_open:
                    time.sleep(2)  # Give Arduino time to reset
                    success = True
                    print(f"Successfully opened connection to {self.arduino_port}")
                else:
                    error_occurred = True
                    success = False
            except Exception as e:
                print(f"Error connecting to Arduino: {e}")
                error_occurred = True
                success = False
        
        if error_occurred:
            self.arduino = None
        
        return success
    
    def check_and_reconnect_arduino(self):
        """
        Check if Arduino is still connected on current port,
        if not, scan all ports and reconnect
        """
        # First check if current connection is still valid
        if self.arduino and hasattr(self.arduino, 'is_open'):
            try:
                if self.arduino.is_open:
                    # Try to check if port still exists
                    if os.path.exists(self.arduino_port):
                        return True
                    else:
                        print(f"Arduino port {self.arduino_port} no longer exists")
            except:
                pass
        
        print("Arduino connection lost - attempting to reconnect...")
        
        # Close any existing connection
        if self.arduino:
            try:
                self.arduino.close()
            except:
                pass
            self.arduino = None
        
        # Search for Arduino on any port
        found_port = self.find_arduino_port()
        
        if found_port:
            self.arduino_port = found_port
            if self.connect_arduino():
                print(f"Successfully reconnected to Arduino on {self.arduino_port}")
                return True
        
        print("Failed to reconnect to Arduino")
        return False
    
    def send_arduino_command(self, command):
        """
        Send command to Arduino with automatic reconnection
        Checks connection before every send
        """
        # Always check and reconnect before sending command
        if not self.check_and_reconnect_arduino():
            print(f"Cannot send command '{command}' - Arduino not available")
            return False
        
        try:
            # Send the command
            command_bytes = command.encode()
            bytes_written = self.arduino.write(command_bytes)
            
            if bytes_written > 0:
                print(f"Sent '{command}' to Arduino on {self.arduino_port}")
                time.sleep(0.1)
                return True
            else:
                print(f"Failed to write command '{command}'")
                return False
                
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            return False
    
    # =====================================================================
    # Server connection and original control code
    # =====================================================================            
    def start(self):
        # Server creation
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        print(f"Host listening on port {self.port}")
        
        # Begin the connection handling thread
        connection_thread = threading.Thread(target = self.handle_connections)
        connection_thread.daemon = True
        connection_thread.start()
        
        # Handle the user input which will be sent
        while self.running:
            message = input("Enter message to send (or 'quit'): \n")
            if message.lower() == 'quit':
                self.running = False
                break
            else:
                self.send_message(message)
                
        # Cleanup
        if self.client_socket:
            self.client_socket.close()
        if self.arduino and hasattr(self.arduino, 'is_open') and self.arduino.is_open:
            self.arduino.close()
        self.server_socket.close()
        
    def handle_connections(self):
        # Continuous client connection control
        while self.running:
            print("Waiting for client connection...")
            
            self.client_socket, addr = self.server_socket.accept()
            self.client_connected = True
            print(f"Client connected from {addr}")
            
            # Begin the listening thread
            listen_thread = threading.Thread(target = self.listen)
            listen_thread.daemon = True
            listen_thread.start()
            
            # wait for client to disconnect before acceptung the next
            listen_thread.join()
            
            # Client dosconnected, clean up
            self.client_connected = False
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            print("Client disconnected. Ready for reconnection...")
            
    def listen(self):
        while self.running and self.client_connected:
            data = self.client_socket.recv(1024).decode('utf-8')
            if data:
                print(f"Received: {data}")
                if data == 'Start':
                    self.send_message("Plant")
                elif data.isdigit():
                    print("Starting integration sequence")
                    self.plant_count = int(data)
                    # Create the instance for the run
                    self.integration = Integration(progress_callback=self.report_progress)
                    
                    # Execute the full pipeline
                    success = self.run_pipeline_sequence()
                    #success = True
                    if success:
                        # Process the data
                        process_success = self.process_data()
                        if process_success:
                            # Transfer all thge files to the client
                            transfer_success = self.send_all_plant_files(self.plant_count - 1)
                            time.sleep(1)
                        else:
                            # Send Failed message to the client
                            self.send_message("Failed")
                            print("Processing failed - Sent the Failed command to client")    
                    else:
                        # Send Failed message to the client
                        self.send_message("Failed")
                        print("Capture failed - Sent the Failed command to client")
                elif data == "Received":
                    # Send END_TRANSFER signal
                    time.sleep(1)
                    self.send_message("END_TRANSFER")
                elif data == "Transfer Complete" or data == "Transfer Failed":
                    if data == "Transfer Complete":
                        # Send the complete message for client
                        self.send_message("Complete")
                        print("Processing and transfer complete - Complete sent to client")
                    else:
                        self.send_message("Failed")
                        print("File transfer failed - send Failed command to the client")
                else:
                    print(f"Received: {data}")
                    print("Enter message to send (or 'quit'): \n")
            else:
                self.client_connected = False
                break
                    
    def run_pipeline_sequence(self):
        self.report_progress("Starting capture sequence...", 0)
        print("============= Capture sequence commence =====================")
        
        # Capture 0 and 180 degrees
        self.report_progress("Capturing 0° and 180° views...", 10)
        capture_success = self.integration.capture(["0_degrees","180_degrees"], self.plant_count)
        
        if not capture_success:
            print("Capture failed for 0 and 100 degrees")
            self.report_progress("Capture failed", -1)
            return False
        
        print("0 and 180 degrees completed")
        self.report_progress("Rotating arm 90°...", 25)
        
        # Rotate anti clockwise 90 degrees
        # Check and reconnect before 'f' command
        if not self.send_arduino_command('f'):
            print("Failed to send 'f' command")
            self.report_progress("Rotation failed", -1)
            return False
        time.sleep(8)
        # Check and reconnect before 's' command
        if not self.send_arduino_command('s'):
            print("Failed to send 's' command")
            self.report_progress("Stop command failed", -1)
            return False
            
        print("Rotation complete")
        self.report_progress("Capturing 90° and 270° views...", 35)
        
        # Capture 90 and 270 degrees
        capture_success = self.integration.capture(["90_degrees","270_degrees"], self.plant_count)
        if not capture_success:
            print("Capture failed for 90 and 270 degrees")
            self.report_progress("Capture failed", -1)
            return False
            
        print("90 and 270 degrees completed")
        self.report_progress("Rotating back to home...", 45)
        
        # Rotate back to homebase
        # Check and reconnect before 'r' command
        if not self.send_arduino_command('r'):
            print("Failed to send 'r' command")
            self.report_progress("Rotation failed", -1)
            return False
        time.sleep(8)
        # Check and reconnect before 's' command
        if not self.send_arduino_command('s'):
            print("Failed to send 's' command")
            self.report_progress("Stop command failed", -1)
            return False
            
        print("Rotation complete")
        print("Full Capture process completed Successfully")
        self.report_progress("Capture sequence complete!", 50)
        
        return True
   
    def process_data(self):
        # Process the captured data using integration class
        self.report_progress("Starting data processing...", 0)
        print("================ Commencing the processing pipeline ==============")
        self.integration.start(self.plant_count)
        self.plant_count += 1
        
        self.report_progress("Processing complete!", 100)
        print("================= Pipeline completed ======================")
        
        return True
                    
    def send_message(self, message):
        if self.client_socket and self.client_connected:
            self.client_socket.send(message.encode('utf-8'))
        else:
            print("No client connected.")
            
if __name__ == "__main__":
    host = Host()
    host.start()
