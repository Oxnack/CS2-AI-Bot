import mss
import cv2
import numpy as np
from PIL import Image
import csv
import time
import os
import random
from pynput import keyboard, mouse
import threading

KEY_MAPPING = {
    'w': 0, 'a': 1, 's': 2, 'd': 3,
    'e': 4, 'r': 5, 
    'shift': 6, 'ctrl': 7,
    '1': 8, '2': 9, '3': 10, '4': 11,
    'space': 12,
    'mouse_left': 13, 'mouse_right': 14
}

class KeyTracker:
    def __init__(self):
        self.keys_pressed = set()
        self.listener = None
        self.lock = threading.Lock()
        
    def on_press(self, key):
        try:
            key_char = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                key_char = 'space'
            elif key == keyboard.Key.shift:
                key_char = 'shift'
            elif key == keyboard.Key.ctrl:
                key_char = 'ctrl'
            else:
                return
        
        with self.lock:
            if key_char in KEY_MAPPING and key_char not in self.keys_pressed:
                self.keys_pressed.add(key_char)
                print(f"Key PRESSED: {key_char}")
    
    def on_release(self, key):
        try:
            key_char = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                key_char = 'space'
            elif key == keyboard.Key.shift:
                key_char = 'shift'
            elif key == keyboard.Key.ctrl:
                key_char = 'ctrl'
            else:
                return
        
        with self.lock:
            if key_char in self.keys_pressed:
                self.keys_pressed.discard(key_char)
                print(f"Key RELEASED: {key_char}")
    
    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
    
    def get_keys_array(self):
        with self.lock:
            return [1 if key in self.keys_pressed else 0 for key in KEY_MAPPING.keys()]

class MouseTracker:
    def __init__(self):
        self.mouse_delta = [0, 0]
        self.mouse_buttons = [0, 0]
        self.last_pos = None
        self.listener = None
        self.lock = threading.Lock()
        
    def on_move(self, x, y):
        with self.lock:
            if self.last_pos is None:
                self.last_pos = (x, y)
                return
                
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            self.mouse_delta = [dx, dy]
            self.last_pos = (x, y)
    
    def on_click(self, x, y, button, pressed):
        with self.lock:
            if button == mouse.Button.left:
                self.mouse_buttons[0] = 1 if pressed else 0
                print(f"Mouse LEFT: {'PRESSED' if pressed else 'RELEASED'}")
            elif button == mouse.Button.right:
                self.mouse_buttons[1] = 1 if pressed else 0
                print(f"Mouse RIGHT: {'PRESSED' if pressed else 'RELEASED'}")
    
    def start(self):
        self.listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.listener.start()
    
    def get_mouse_data(self):
        with self.lock:
            data = self.mouse_buttons + self.mouse_delta
            self.mouse_delta = [0, 0]
            return data

def capture_game_window():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return sct.grab(monitor)

def process_frame(frame, output_dir):
    img = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
    img = img.resize((227, 128))
    img = img.convert('L')
    
    filename = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8)) + '.jpg'
    filepath = os.path.join(output_dir, filename)
    img.save(filepath, 'JPEG', quality=85)
    
    return filename

class CSGODataCollector:
    def __init__(self, output_dir="dataset"):
        self.key_tracker = KeyTracker()
        self.mouse_tracker = MouseTracker()
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.csv_file = None
        self.csv_writer = None
        self.running = False
        
    def setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
    
    def setup_csv(self):
        csv_path = os.path.join(self.output_dir, "dataset.csv")
        file_exists = os.path.isfile(csv_path)
        
        self.csv_file = open(csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            header = ['filename'] + list(KEY_MAPPING.keys()) + ['mouse_left', 'mouse_right', 'mouse_dx', 'mouse_dy']
            self.csv_writer.writerow(header)
    
    def start(self):
        self.setup_directories()
        self.setup_csv()
        
        self.key_tracker.start()
        self.mouse_tracker.start()
        
        self.running = True
        print("Data collection STARTED")
        print("Press Ctrl+C to stop")
        
        frame_count = 0
        try:
            while self.running:
                start_time = time.time()
                
                frame = capture_game_window()
                filename = process_frame(frame, self.images_dir)
                
                keys_array = self.key_tracker.get_keys_array()
                mouse_data = self.mouse_tracker.get_mouse_data()
                
                row = [filename] + keys_array + mouse_data
                self.csv_writer.writerow(row)
                
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"Collected {frame_count} frames...")
                
                elapsed = time.time() - start_time
                sleep_time = max(0.1 - elapsed, 0)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.csv_file:
            self.csv_file.close()
        print("Data collection STOPPED")

if __name__ == "__main__":
    collector = CSGODataCollector()
    collector.start()