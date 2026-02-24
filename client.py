"""
GUI client that communicates with the Jetson Nano host server over TCP.
Sends control commands and receives progress updates and result files.
Runs on any machine with a network connection to the Jetson Nano.
"""

import socket
import threading
import serial
import time
import os

class Client:
	def __init__(self, server_ip,port = 8888, arduino_port = '/dev/ttyACM0', baud_rate = 9600):
		self.server_ip = server_ip
		self.port = port
		self.socket = None
		self.running = True
					
	def connect(self):
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.connect((self.server_ip, self.port))
		print(f"Connected to host at {self.server_ip}:{self.port}")
		
		# Start listening thread
		listen_thread = threading.Thread(target = self.listen)
		listen_thread.daemon = True
		listen_thread.start()
		
		# Handle the users input message
		while self.running:
			message = input("Enter message to send (or 'quit'): \n")
			if message.lower() == 'quit':
				self.running = False
				break
			else:
				self.send_message(message)
			
	def listen(self):
		while self.running:
			if self.socket:
				data = self.socket.recv(1024).decode('utf-8')
				if data:
					print(f"\nReceived: {data}")
					if data == 'Step 1 Complete':
						self.send_arduino_command('f')
						time.sleep(8)
						self.send_arduino_command('s')
						self.send_message("next")
					elif data == 'Capture Complete':
						self.send_arduino_command('r')
						time.sleep(8)
						self.send_arduino_command('s')
						print("Done")
				else:
					print("\nHost disconnected")
					break
					
	def send_message(self, message):
		if self.socket:
			self.socket.send(message.encode('utf-8'))
			
if __name__ == "__main__":
	server_ip = input("Enter host IP: ")
	client = Client(server_ip)
	client.connect()
