from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import win32con
import glfw
import cv2
import numpy as np
from collections import deque
import math
import json
import socket
import threading
import bettercam
import torch
import win32api
import keyboard
import ctypes
import os
import time
import imgui
from OpenGL.GL import glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_NEAREST
from termcolor import colored
from ultralytics import YOLO
from pynput import mouse
import random

productname = "AimFlow"

os.system('cls')
print(f"{productname} Aimbot Module loaded")
print(f"Initializing {productname}...")
time.sleep(1)

def clear():
    os.system('cls')

username = False
lastlogin = False
hwid = False
exp = False
chtstatus = False
tmssent = False
intrdy = False

def show_error_message(message, title='Error'):
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)

def ldrinf():
    global username, lastlogin, hwid, exp, tmssent, chtstatus, intrdy
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('localhost', 9999))
            received_data = client_socket.recv(1024).decode()
            data = json.loads(received_data)
            username = data.get("Username")
            lastlogin = data.get("Lastlogin")
            hwid = data.get("Hwid")
            exp = data.get("Expiration")
            chtstatus = data.get("Status")
            tmssent = data.get("Timesrecieved")
            intrdy = True
        except Exception:
            os._exit(1)
        if "y" in chtstatus:
            pass
        elif "n" in chtstatus:
            clear()
            show_error_message(productname + " is down at this moment.")
            time.sleep(3)
            os._exit(1)
        elif "p" in chtstatus:
            clear()
            if os.path.exists("C:\\AimFlow\\lic.data"):
                os.remove("C:\\AimFlow\\lic.data")
            show_error_message(productname + " is down at this moment.")
            time.sleep(3)
            os._exit(1)
threading.Thread(target=ldrinf, daemon=True).start()

while True:
    if intrdy == False:
        pass
    else:
        break

clear()
print("Username from Loader > " + username)
print("Last Login from Loader > " + lastlogin)
print("HWID from Loader > " + hwid)
print("Expiration Date from Loader > " + exp)
print("Cheat Status from Loader > " + chtstatus)
print("Times Sent from Loader > " + tmssent)

time.sleep(10)
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# Systém přihlášení

config_path = os.path.join(f"C:/{productname}/modules/config.json")

class Aimgod:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    pixel_increment = 1
    aimgod_status = colored("ENABLED", 'green')

    def __init__(self, box_constant=350, collect_data=False, mouse_delay=0.0001):
        self.box_constant = box_constant
        self.detection_width = 1280
        self.detection_height = 720
        clear()
        print("[+] Loading the neural network model")
        time.sleep(1.5)
        if os.path.exists(f"C:/{productname}/modules/model.pt"):
            self.model = YOLO(f"C:/{productname}/modules/model.pt")
        else:
            clear()
            print(colored("[-] Model not found", "red"))
            time.sleep(2)
            os._exit(1)
        if os.path.exists(f"C:/{productname}/modules/model.pt"):
            os.system(f'attrib -h -s -r C:/{productname}/modules/model.pt')
        else:
            pass
        os.remove(f"C:/{productname}/modules/model.pt")

        if torch.cuda.is_available():
            clear()
            print(colored("[+] CUDA ACCELERATION ENABLED", "green"))
            time.sleep(3)
            clear()
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), win32con.SW_HIDE)
        else:
            print(colored("[-] CUDA ACCELERATION IS UNAVAILABLE", "red"))
            while True:
                kundak = input(colored("Choose from >>\n (1) Install Cuda Modules\n (2) Skip Cuda Modules [more cpu usage]\n>> ", "red"))
                if kundak == "1":
                    clear()
                    print(colored("Installing Cude Modules... [this might take some time]", "blue"))
                    try:
                        os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 <NUL")
                        clear()
                        print(colored("Cuda Modules installed successfully", "green"))
                        time.sleep(2)
                    except Exception:
                        clear()
                        print(colored("Failed to install Cuda Modules...\nSwitching to CPU Usage", "red"))
                        time.sleep(2)
                elif kundak == "2":
                    clear()
                    break
                else:
                    print(colored("[-] Invalid option", "red"))
                    time.sleep(2)
                    clear()
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), win32con.SW_HIDE)

        self.load_config()
        self.collect_data = collect_data
        self.mouse_delay = mouse_delay
        self.camera = None
        self.center_x = self.box_constant // 2
        self.center_y = self.box_constant // 2
        self.target_lost_frames = 0
        self.max_lost_frames = 20
        self.last_target = None
        self.debug_mode = False
        self.max_detections = 5


        self.min_detection_frames = 3  
        self.current_detection_frames = 0
        self.last_detection_coords = None
        self.max_aim_distance = 400  
        self.min_frame_time = 1/120  
        self.last_aim_time = time.time()
        
  
        self.auto_fire_enabled = False
        self.auto_fire_key = None
        self.auto_fire_cooldown = 0.1 
        self.alignment_threshold = 8.0
        self.last_auto_fire = 0
        self.mouse_controller = mouse.Controller()
        

        self.target_lock_threshold = 5 
        self.min_target_size = 3 
        self.target_lock = None  
        self.lock_frames = 0
        self.min_lock_frames = 2 
        self.max_target_switch_distance = 100 
        self.aim_height = "torso"
        self.confidence_threshold = 0.5
        self.dynamic_iou = 0.8
        self.smoothing = 1.5
        self.aim_speed = 1.0
        self.aim_key = 0x02  # Default aim key

    def load_config(self):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.smoothing = config.get("smoothing", 0.0)
                self.aim_speed = config.get("aim_speed", 1.0)
                self.target_threshold = config.get("target_threshold", 3.5)
                self.confidence_threshold = 0.5
                self.aim_height = config.get("aim_height", "upper_chest")
                self.max_detections = config.get("max_detections", 5)
                
                aim_key_value = config.get("aim_key", "0x02")
                if isinstance(aim_key_value, str):
                    self.aim_key = int(aim_key_value, 16)
                else:
                    self.aim_key = int("0x02", 16)
                    

                self.auto_fire_key = int(config.get('auto_fire_key', '0x75'), 16)
                self.auto_fire_cooldown = config.get('auto_fire_cooldown', 0.2)
                self.alignment_threshold = config.get('alignment_threshold', 5)

    def aim_height_adjustment(self, x, y):
        """
        Enhanced height adjustment for close combat
        """
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Close range adjustments
        if distance < 50:

            return y - 15  
        elif distance < 100:

            return y - 25
        else:

            return y - 35

    def start_capture(self):

        primary_monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(primary_monitor)
        screen_res_x = video_mode.size.width
        screen_res_y = video_mode.size.height

        half_screen_width = screen_res_x / 2
        half_screen_height = screen_res_y / 2
        detection_box = {
            'left': int(half_screen_width - self.box_constant // 2),
            'top': int(half_screen_height - self.box_constant // 2),
            'width': int(self.box_constant),
            'height': int(self.box_constant)
        }

        self.camera = bettercam.create()
        self.camera.start(region=(
            detection_box['left'],
            detection_box['top'],
            detection_box['left'] + detection_box['width'],
            detection_box['top'] + detection_box['height']
        ))
        print("[+] Screen capture started.")

    def process_frame(self):
        if self.camera is None:
            return

        frame_start_time = time.time()
        frame = self.camera.get_latest_frame()
        if frame is None:
            return


        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        debug_frame = frame.copy() if self.debug_mode else None

        if torch.cuda.is_available():
            device_name = "cuda:0"
        else:
            device_name = "cpu"

        results = self.model.predict(frame, conf=self.confidence_threshold, iou=self.dynamic_iou, device=device_name, verbose=False)

        if results is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.cpu() 
            
  
            center_frame = np.array([self.center_x, self.center_y])
            distances = []
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy() 
                box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                distance = np.linalg.norm(box_center - center_frame)
                distances.append((distance, box))
                
                
                if self.debug_mode:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    confidence = box.conf.numpy().item()
                    
                   
                    if confidence > 0.7:
                        color = (0, 255, 0)  
                    elif confidence > 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)  
                    
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(debug_frame, conf_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            distances.sort(key=lambda x: x[0])
            if distances:
                best_box = distances[0][1]
                if best_box.conf.numpy() > self.confidence_threshold:
                    x1, y1, x2, y2 = best_box.xyxy[0].numpy()
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    current_coords = (x, y) 
        
                
                    if self.last_detection_coords:
                        distance = np.sqrt((current_coords[0] - self.last_detection_coords[0])**2 + 
                                         (current_coords[1] - self.last_detection_coords[1])**2)
                        
                        if distance < 50: 
                            self.current_detection_frames += 1
                        else:
                            self.current_detection_frames = 0  
                            
                    self.last_detection_coords = current_coords


                    if self.current_detection_frames >= self.min_detection_frames:
                     
                        self.move_crosshair(x, y)
                    self.target_lost_frames = 0
                    
                    if self.debug_mode:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        
                   
                    if self.auto_fire_enabled:
                        self.check_auto_fire([x1, y1, x2, y2], best_box.conf.numpy())
                else:
                    self.target_lost_frames += 1
            else:
                self.target_lost_frames += 1

        if self.target_lost_frames > self.max_lost_frames:
            self.last_target = None


        try:
            if self.debug_mode and debug_frame is not None:
                frame_time = time.time() - frame_start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(debug_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
           
                cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
                cv2.imshow("Debug", debug_frame)
                cv2.waitKey(1)
            elif not self.debug_mode:
                try:
                    cv2.destroyWindow("Debug")
                except:
                    pass
        except Exception as e:
            print(f"Debug window error: {e}")
            time.sleep(20)

    def move_crosshair(self, x, y):
        """Smooth, stable aim transitions at high speeds"""
        if not self.is_targeted():
            return

        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)

        smoothing = self.get_dynamic_smoothing(distance)
        step_x = dx * (1.0 - smoothing)
        step_y = dy * (1.0 - smoothing)


        if abs(step_x) < 0.5 and abs(step_y) < 0.5:
            return

        move_x = int(step_x)
        move_y = int(step_y)

        self.ii_.mi = MouseInput(move_x, move_y, 0, 0x0001, 0, ctypes.pointer(self.extra))
        input_obj = Input(ctypes.c_ulong(0), self.ii_)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))

    def get_dynamic_smoothing(self, distance):
        """
        More aggressive smoothing to reduce jitter at high speeds
        """
        if distance < 15:
            return 0.25
        elif distance < 30:
            return 0.32
        elif distance < 50:
            return 0.38
        else:
            return 0.45 + (distance * 0.001)

    def aim_at_target(self, x, y):
        """
        Enhanced aiming system with stable lock-on and reduced jitter
        """
        if not self.is_targeted():
            return

        # Calculate distance from crosshair to target
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)

        lock_radius = 5.0
        

        if distance <= lock_radius:

            return
            
        
        
        smoothing = self.get_dynamic_smoothing(distance)


        move_x = int(dx * (1.0 - smoothing))
        move_y = int(dy * (1.0 - smoothing))


        if abs(move_x) < 1 and abs(move_y) < 1:
            return


        self.ii_.mi = MouseInput(
            move_x, move_y,
            0, 0x0001, 0,
            ctypes.pointer(self.extra)
        )
        input_obj = Input(ctypes.c_ulong(0), self.ii_)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))

    @staticmethod
    def sleep(duration, get_now=time.perf_counter):
        if duration == 0:
            return
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()

    def is_targeted(self):
        # Adjusted handling for X1
        if self.aim_key == 0x05:  
            return bool(win32api.GetAsyncKeyState(0x05) & 0x8000)
        elif self.aim_key == 0x06:  # X2 button
            return bool(win32api.GetAsyncKeyState(0x06) & 0x8000)
        else:
            return bool(win32api.GetAsyncKeyState(self.aim_key) & 0x8000)

    def aim_at_target(self, target_pos, current_pos, smoothing=0.5):
        """
        Smooth aiming system with consistent target tracking
        
        Args:
            target_pos: (x, y) tuple of target position
            current_pos: (x, y) tuple of current aim position
            smoothing: float between 0-1, higher = smoother movement
        """
        try:
            # Validate inputs
            if not all(isinstance(pos, tuple) and len(pos) == 2 
                      for pos in (target_pos, current_pos)):
                raise ValueError("Invalid position format")

            # Calculate distance to target
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            # Skip if target too far
            if distance > self.max_aim_distance:
                return current_pos

            # Apply smoothing
            smoothed_x = current_pos[0] + (dx * (1.0 - smoothing))
            smoothed_y = current_pos[1] + (dy * (1.0 - smoothing))

            # Remove the random offset lines for reduced shake:
            final_x = smoothed_x
            final_y = smoothed_y

            # Frame timing control
            current_time = time.time()
            if hasattr(self, 'last_aim_time'):
                frame_delta = current_time - self.last_aim_time
                if frame_delta < self.min_frame_time:
                    time.sleep(self.min_frame_time - frame_delta)
            print(f"Aiming error: {str(e)}")

            return (final_x, final_y)

        except Exception as e:
            print(f"Aiming error: {str(e)}")
            time.sleep(20)
            return current_pos

    def handle_input(self):

        
        if keyboard.is_pressed(self.auto_fire_key):
            self.toggle_auto_fire()

    def check_auto_fire(self, target_bbox, confidence):
        """Enhanced auto-fire with range-specific behavior"""
        if not (self.is_targeted() and self.auto_fire_enabled):
            return

        target_center_x = (target_bbox[0] + target_bbox[2]) / 2
        target_center_y = (target_bbox[1] + target_bbox[3]) / 2
        dx = abs(target_center_x - self.center_x)
        dy = abs(target_center_y - self.center_y)
        distance = math.sqrt(dx**2 + dy**2)

        # Scale alignment threshold for long distances
        dynamic_thresh = self.alignment_threshold - min(distance * 0.02, self.alignment_threshold - 1)

        # Increasing confidence requirement for long distances
        dynamic_conf = max(0.3, min(1.0, confidence + (distance * 0.001)))

        if distance <= 30:  # close range
            if dx <= 20 and dy <= 20:
                self.mouse_controller.click(mouse.Button.left)
        elif distance <= 80:  # mid range
            if dx <= dynamic_thresh * 1.2 and dy <= dynamic_thresh * 1.2 and dynamic_conf >= 0.4:
                self.mouse_controller.click(mouse.Button.left)
        else:  # long range
            if dx <= dynamic_thresh and dy <= dynamic_thresh and dynamic_conf >= 0.5:
                time.sleep(0.05)  # small delay for stability
                self.mouse_controller.click(mouse.Button.left)

    def compensate_recoil(self):
        """Advanced recoil compensation with distance-based patterns"""
        # Get current target distance if available
        distance = getattr(self, 'last_target_distance', 100)
        
        # Scale recoil based on distance
        distance_factor = max(0.5, min(1.5, distance / 100))
        
        # Base recoil patterns
        if distance < 30:  # Close range
            recoil_x = random.uniform(-3, 3) * distance_factor
            recoil_y = random.uniform(-4, -2) * distance_factor
        elif distance < 60:  # Medium range
            recoil_x = random.uniform(-2, 2) * distance_factor
            recoil_y = random.uniform(-3, -1) * distance_factor
        else:  # Long range
            recoil_x = random.uniform(-1, 1) * distance_factor
            recoil_y = random.uniform(-2, -0.5) * distance_factor

        # Apply recoil movement
        self.ii_.mi = MouseInput(
            int(recoil_x), int(recoil_y),
            0, 0x0001, 0, 
            ctypes.pointer(self.extra)
        )
        input_obj = Input(ctypes.c_ulong(0), self.ii_)
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))
        
        # Add small delay for realism
        time.sleep(0.001)

    def has_clear_shot(self, target_x, target_y):
        """Check if there's a clear line of sight to target"""
        # Get line between crosshair and target
        points = self.get_line_points(self.center_x, self.center_y, target_x, target_y)
        
        # Sample points along line
        for x, y in points:
            if not self.is_valid_target_pixel(x, y):
                return False
        return True

    def get_line_points(self, x1, y1, x2, y2):
        """Get points along line between two coordinates"""
        points = []
        # Convert numpy floats to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = int(1 + dx + dy)  # Convert to integer for range()
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            points.append((int(x), int(y)))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return points

    def is_valid_target_pixel(self, x, y):
        """Check if pixel is part of valid target"""
        # Get pixel color/intensity
        try:
            pixel = self.current_frame[y, x]
            # Avoid shooting at very dark (walls) or very bright (sky) areas
            return 30 < np.mean(pixel) < 220
        except:
            return False

    def get_best_target(self, boxes):
        """Get closest valid target to crosshair"""
        valid_targets = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate distance to crosshair
            dx = center_x - self.center_x
            dy = center_y - self.center_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= self.max_target_switch_distance:
                valid_targets.append((distance, box))
                
        if valid_targets:
            valid_targets.sort(key=lambda x: x[0])  # Sort by distance
            return valid_targets[0][1]  # Return closest target
            
        return None

    def is_valid_target(self, x1, y1, x2, y2, confidence):
        """Validate target size and position"""
        width = x2 - x1
        height = y2 - y1
        
        # Size check
        if width < self.min_target_size or height < self.min_target_size:
            return False
            
        # Position check
        if not (0 <= x1 < self.box_constant and 0 <= y1 < self.box_constant):
            return False
            
        # Confidence check
        if confidence < self.confidence_threshold:
            return False
            
        return True

    def update_target_lock(self, target_x, target_y):
        """Update target lock status"""
        if self.target_lock is None:
            self.target_lock = (target_x, target_y)
            self.lock_frames = 1
            return False
            
        # Check if target moved too far
        dx = target_x - self.target_lock[0]
        dy = target_y - self.target_lock[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > self.target_lock_threshold:
            self.target_lock = (target_x, target_y)
            self.lock_frames = 1
            return False
            
        # Update lock
        self.target_lock = (target_x, target_y)
        self.lock_frames += 1
        
        return self.lock_frames >= self.min_lock_frames

    def clear_target_lock(self):
        """Clear target lock when no valid target"""
        self.target_lock = None
        self.lock_frames = 0

def set_theme():
    style = imgui.get_style()
    
    # Set dark modern theme colors
    style.colors[imgui.COLOR_TEXT] = (1.00, 1.00, 1.00, 1.00)
    style.colors[imgui.COLOR_TEXT_DISABLED] = (0.50, 0.50, 0.50, 1.00)
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.06, 0.94)
    style.colors[imgui.COLOR_CHILD_BACKGROUND] = (0.08, 0.08, 0.08, 0.94)
    style.colors[imgui.COLOR_POPUP_BACKGROUND] = (0.08, 0.08, 0.08, 0.94)
    style.colors[imgui.COLOR_BORDER] = (0.43, 0.43, 0.50, 0.50)
    style.colors[imgui.COLOR_BORDER_SHADOW] = (0.00, 0.00, 0.00, 0.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.12, 0.12, 0.12, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.25, 0.25, 0.25, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.30, 0.30, 0.30, 1.00)
    style.colors[imgui.COLOR_BUTTON] = (0.15, 0.15, 0.15, 1.00)
    style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.25, 0.25, 0.25, 1.00)
    style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.30, 0.30, 0.30, 1.00)
    
    # Set modern style properties
    style.window_padding = (15.0, 15.0)
    style.frame_padding = (5.0, 5.0)
    style.item_spacing = (12.0, 8.0)
    style.item_inner_spacing = (8.0, 6.0)
    style.window_rounding = 5.0
    style.frame_rounding = 4.0
    style.popup_rounding = 4.0
    style.grab_rounding = 4.0

def set_professional_theme():
    style = imgui.get_style()
    
    # Set a modern dark theme
    style.colors[imgui.COLOR_TEXT] = (1.00, 1.00, 1.00, 1.00)
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.10, 0.10, 0.10, 1.00)
    style.colors[imgui.COLOR_CHILD_BACKGROUND] = (0.15, 0.15, 0.15, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.20, 0.20, 0.20, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.25, 0.25, 0.25, 1.00)
    style.colors[imgui.COLOR_BUTTON] = (0.25, 0.25, 0.25, 1.00)
    style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.35, 0.35, 0.35, 1.00)
    style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.45, 0.45, 0.45, 1.00)
    
    # Set padding and rounding
    style.window_padding = (10.0, 10.0)
    style.frame_padding = (5.0, 5.0)
    style.item_spacing = (10.0, 10.0)
    style.window_rounding = 5.0
    style.frame_rounding = 3.0
    style.popup_rounding = 3.0
    style.grab_rounding = 3.0

def set_modern_theme():
    style = imgui.get_style()
    
    # Dark theme base colors
    style.colors[imgui.COLOR_TEXT] = (0.95, 0.96, 0.98, 1.00)
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.11, 0.15, 0.17, 1.00)
    style.colors[imgui.COLOR_CHILD_BACKGROUND] = (0.15, 0.18, 0.22, 1.00)
    style.colors[imgui.COLOR_BORDER] = (0.08, 0.10, 0.12, 1.00)
    
    # Accent colors
    style.colors[imgui.COLOR_HEADER] = (0.20, 0.25, 0.29, 1.00)
    style.colors[imgui.COLOR_HEADER_HOVERED] = (0.26, 0.59, 0.98, 0.80)
    style.colors[imgui.COLOR_HEADER_ACTIVE] = (0.26, 0.59, 0.98, 1.00)
    
    # Interactive elements
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.20, 0.25, 0.29, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.25, 0.30, 0.35, 1.00)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.30, 0.35, 0.40, 1.00)
    style.colors[imgui.COLOR_BUTTON] = (0.20, 0.25, 0.29, 1.00)
    style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.25, 0.30, 0.35, 1.00)
    style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.30, 0.35, 0.40, 1.00)
    
    # Sliders/Checkboxes
    style.colors[imgui.COLOR_SLIDER_GRAB] = (0.26, 0.59, 0.98, 0.80)
    style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.26, 0.59, 0.98, 1.00)
    style.colors[imgui.COLOR_CHECK_MARK] = (0.26, 0.59, 0.98, 1.00)
    
    # Window styling
    style.window_padding = (15, 15)
    style.frame_padding = (5, 5)
    style.item_spacing = (12, 8)
    style.item_inner_spacing = (8, 6)
    style.indent_spacing = 25.0
    style.scrollbar_size = 15.0
    style.grab_min_size = 5.0
    
    # Rounded corners
    style.window_rounding = 6.0
    style.frame_rounding = 4.0
    style.popup_rounding = 4.0
    style.scrollbar_rounding = 9.0
    style.grab_rounding = 3.0
    style.tab_rounding = 4.0

def render_main_window(aimgod, window_width, window_height):
    imgui.set_next_window_size(window_width, window_height)
    imgui.set_next_window_position(0, 0)
    
    flags = (
        imgui.WINDOW_NO_TITLE_BAR | 
        imgui.WINDOW_NO_RESIZE | 
        imgui.WINDOW_NO_MOVE |
        imgui.WINDOW_NO_COLLAPSE
    )

    imgui.begin("AimFlow", flags=flags)
    
    # Header
    header_font = imgui.get_io().fonts.add_font_from_file_ttf("c:\Windows\Fonts\ARIBLK.TTF", 20)
    imgui.push_font(header_font)
    imgui.text("AimFlow")
    imgui.pop_font()
    imgui.separator()
    
    # Two-column layout
    imgui.columns(2)
    imgui.set_column_width(0, window_width * 0.25)
    
    # Left panel
    render_left_panel(aimgod)
    
    imgui.next_column()
    
    # Right panel
    render_right_panel(aimgod)
    
    imgui.columns(1)
    imgui.end()

def render_left_panel(aimgod):
    if imgui.begin_child("left_panel", 0, 0, False):
        # Main controls
        imgui.text("Aim Controls")
        imgui.separator()
        
        changed, aimgod.enabled = imgui.checkbox("Enable Aimbot", aimgod.enabled)
        changed, aimgod.auto_fire_enabled = imgui.checkbox("Auto-Fire", aimgod.auto_fire_enabled)
        
        imgui.push_item_width(imgui.get_window_width() * 0.8)
        changed, aimgod.smoothing = imgui.slider_float(
            "Smoothing", aimgod.smoothing, 0.0, 1.0, "%.2f"
        )
        changed, aimgod.aim_speed = imgui.slider_float(
            "Aim Speed", aimgod.aim_speed, 0.1, 2.5, "%.2f"
        )
        imgui.pop_item_width()
    
    imgui.end_child()

def render_right_panel(aimgod):
    if imgui.begin_child("right_panel", 0, 0, False):
        if imgui.begin_tab_bar("##tabs"):
            
            # Settings tab
            if imgui.begin_tab_item("Settings")[0]:
                render_settings_tab(aimgod)
                imgui.end_tab_item()
            
            # Config tab    
            if imgui.begin_tab_item("Config")[0]:
                render_config_tab(aimgod)
                imgui.end_tab_item()
                
            imgui.end_tab_bar()
            
    imgui.end_child()

def render_settings_tab(aimgod):
    imgui.text("Settings")
    imgui.separator()
    imgui.text("Adjust your settings here.")
    # Add more settings controls as needed

def render_config_tab(aimgod):
    imgui.text("Configuration Settings")
    imgui.separator()
    imgui.text("Mouse Key:")
    changed, aim_key_input = imgui.input_text("Aim Key (Hex)", f"0x{aimgod.aim_key:02x}", 10)
    if changed:
        try:
            aimgod.aim_key = int(aim_key_input, 16)  # Convert hex string to int
        except ValueError:
            pass  # Handle invalid input if necessary

    # Button to update config
    if imgui.button("Update"):
        save_config(aimgod)  # Call the function to save the updated config

def main():
    if not glfw.init():
        return False

    # Set window properties
    glfw.window_hint(glfw.DECORATED, False)  # Remove window decorations
    glfw.window_hint(glfw.RESIZABLE, False)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    window_width = 600  # Adjusted width
    window_height = 400  # Adjusted height
    window_x = (mode.size.width - window_width) // 2
    window_y = (mode.size.height - window_height) // 2
    
    window = glfw.create_window(window_width, window_height, "AimFlow", None, None)
    glfw.set_window_pos(window, window_x, window_y)

    if not window:
        glfw.terminate()
        return False

    glfw.make_context_current(window)

    # Initialize ImGui
    imgui.create_context()
    imgui.set_current_context(imgui.get_current_context())
    impl = GlfwRenderer(window)

    # Set professional theme
    set_professional_theme()

    # Initialize variables
    aimgod = Aimgod()
    aim_height_options = ["Head", "Upper Body", "Lower Body"]
    aim_height_mapping = {
        "Head": "head",
        "Upper Body": "torso",
        "Lower Body": "legs"
    }
    current_aim_height = next(
        (k for k, v in aim_height_mapping.items() if v == aimgod.aim_height), 
        "Upper Body"
    )
    confidence = aimgod.confidence_threshold
    smoothing = aimgod.smoothing
    aim_speed = aimgod.aim_speed
    max_detections = aimgod.max_detections
    debug_mode_checkbox = False  # Checkbox for debug mode
    aimgod_running = False

    # Aim key configuration
    aim_key = aimgod.aim_key  # Default aim key

    # Variables for dragging
    dragging = False
    last_x, last_y = 0, 0

    # Textbox for custom configuration
    custom_config_value = ""
    aim_key_input = "0x"  # Default input for aim key

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        # Handle mouse dragging
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            if not dragging:
                dragging = True
                last_x, last_y = glfw.get_cursor_pos(window)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
            dragging = False

        if dragging:
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - last_x
            dy = current_y - last_y
            
            # Get current window position
            window_x, window_y = glfw.get_window_pos(window)
            
            # Update window position
            glfw.set_window_pos(window, window_x + int(dx), window_y + int(dy))
            last_x, last_y = current_x, current_y

        imgui.new_frame()

        # Set window to cover entire GLFW window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_width, window_height)
        
        window_flags = (
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_TITLE_BAR
        )
        
        imgui.begin("AimFlow", flags=window_flags)

        if imgui.begin_tab_bar("##tabs"):
            # Aimbot tab
            if imgui.begin_tab_item("Aimbot")[0]:
                # Center the content
                content_width = 500
                imgui.set_cursor_pos_x((window_width - content_width) * 0.5)
                
                imgui.begin_child("aimbot_settings", width=content_width, height=300)
                
                imgui.text("Target Settings")
                imgui.separator()
                
                # Add proper item width for controls
                imgui.push_item_width(200)
                
                changed, current_aim_height_index = imgui.combo(
                    "Target Area", 
                    aim_height_options.index(current_aim_height), 
                    aim_height_options
                )
                if changed:
                    current_aim_height = aim_height_options[current_aim_height_index]
                    aimgod.aim_height = aim_height_mapping[current_aim_height]

                imgui.spacing()
                imgui.text("Accuracy Settings")
                imgui.separator()
                
                # Sliders for updating config
                changed, confidence = imgui.slider_float("Confidence", confidence, 0.0, 1.0)
                if changed:
                    aimgod.confidence_threshold = confidence

                changed, smoothing = imgui.slider_float("Smoothing", smoothing, 0.0, 1.0)
                if changed:
                    aimgod.smoothing = smoothing

                changed, aim_speed = imgui.slider_float("Aim Speed", aim_speed, 0.0, 2.5)
                if changed:
                    aimgod.aim_speed = aim_speed

                imgui.pop_item_width()
                
                imgui.spacing()
                imgui.spacing()
                
                # Checkbox for enabling/disabling Aimbot
                changed, aimbot_enabled = imgui.checkbox("Enable Aimbot", aimgod_running)
                if changed:
                    aimgod_running = aimbot_enabled
                    if aimgod_running:
                        aimgod.start_capture()
                    else:
                        pass
                
                imgui.spacing()
                imgui.separator()
                imgui.text("Auto-Fire Settings")
                
                # Add Auto-Fire checkbox
                changed, auto_fire_enabled = imgui.checkbox("Enable Auto-Fire", aimgod.auto_fire_enabled)
                if changed:
                    aimgod.auto_fire_enabled = auto_fire_enabled
                    status = "ENABLED" if auto_fire_enabled else "DISABLED"
                    print(colored(f"[*] Auto-fire {status}", "cyan"))

                # Add Auto-Fire sensitivity slider
                imgui.push_item_width(200)
                changed, alignment = imgui.slider_float("Alignment Threshold", 
                                                      aimgod.alignment_threshold, 
                                                      1.0, 20.0, 
                                                      "%.1f px")
                if changed:
                    aimgod.alignment_threshold = alignment

                # Add Auto-Fire cooldown slider
                changed, cooldown = imgui.slider_float("Fire Cooldown", 
                                                     aimgod.auto_fire_cooldown,
                                                     0.1, 1.0,
                                                     "%.2f sec")
                if changed:
                    aimgod.auto_fire_cooldown = cooldown
                imgui.pop_item_width()
                
                imgui.end_child()
                imgui.end_tab_item()

            # Advanced tab
            if imgui.begin_tab_item("Advanced")[0]:
                imgui.begin_child("advanced_settings", width=600, height=400)
                imgui.text("Advanced Settings")
                imgui.separator()

                # Slider for max detected objects
                changed, max_detections = imgui.slider_int("Max Detected Objects", max_detections, 1, 100)
                if changed:
                    aimgod.max_detections = max_detections

                # Checkbox for debug mode
                changed, debug_mode_checkbox = imgui.checkbox("Enable Debug Window", debug_mode_checkbox)
                if changed:
                    aimgod.debug_mode = debug_mode_checkbox  # Assuming Aimgod has a debug_mode attribute

                imgui.end_child()
                imgui.end_tab_item()
             
            # Config tab
            if imgui.begin_tab_item("Config")[0]:
                imgui.begin_child("config_settings", width=600, height=400)
                imgui.text("Configuration Settings")
                imgui.separator()


                # Mouse key configuration
                imgui.text("Mouse Key:")
                changed, aim_key_input = imgui.input_text("Aim Key (Hex)", f"0x{aimgod.aim_key:02x}", 10)
                if changed:
                    try:
                        aimgod.aim_key = int(aim_key_input, 16)  # Convert hex string to int
                    except ValueError:
                        pass  # Handle invalid input if necessary

                # Button to update config
                if imgui.button("Update"):
                    save_config(aimgod)  # Call the function to save the updated config

                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.end()

        # Rendering
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        # Process aimbot if running
        if aimgod_running:
            aimgod.process_frame()

        # Check for ESC key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    # Cleanup
    impl.shutdown()
    glfw.terminate()

    return True  # Indicate success

def save_config(aimgod):
    """Save config to Desktop AimFlow folder"""
    config = {
        "smoothing": aimgod.smoothing,
        "aim_speed": aimgod.aim_speed, 
        "target_threshold": aimgod.target_threshold,
        "confidence_threshold": aimgod.confidence_threshold,
        "aim_height": aimgod.aim_height,
        "max_detections": aimgod.max_detections,
        "aim_key": f"0x{aimgod.aim_key:02x}",
    }
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            existing_config = json.load(f)
            existing_config.update(config)
            config = existing_config

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()


class EnhancedObjectTracker:
    def __init__(self):
        # Kalman filter initialization 
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1], 
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Target tracking parameters
        self.target_history = deque(maxlen=10)
        self.min_confidence = 0.65
        self.smoothing = 0.7
        self.velocity_threshold = 50
        self.lost_frames = 0
        self.max_lost_frames = 15

    def predict_next_pos(self, detection):
        # Predict using Kalman filter
        self.kalman.predict()
        measurement = np.array([[detection[0]], [detection[1]]], np.float32)
        correction = self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction[:2].reshape(-1)

    def update(self, detection, confidence):
        if confidence < self.min_confidence:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                return None
            return self.get_smoothed_position()

        self.lost_frames = 0
        predicted = self.predict_next_pos(detection)
        
        # Anti-shake filtering
        if self.target_history:
            last_pos = self.target_history[-1]
            velocity = np.linalg.norm(predicted - last_pos)
            
            if velocity > self.velocity_threshold:
                predicted = self.apply_smoothing(predicted, last_pos)
        
        self.target_history.append(predicted)
        return predicted

    def apply_smoothing(self, new_pos, last_pos):
        dx = new_pos[0] - last_pos[0] 
        dy = new_pos[1] - last_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Adaptive smoothing based on distance
        smoothing = min(0.9, self.smoothing * (distance / self.velocity_threshold))
        
        x = last_pos[0] + (1.0 - smoothing) * dx
        y = last_pos[1] + (1.0 - smoothing) * dy
        return np.array([x, y])

    def get_smoothed_position(self):
        if not self.target_history:
            return None
            
        positions = list(self.target_history)
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights /= weights.sum()
        
        return np.average(positions, weights=weights, axis=0)

class DetectionProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.min_target_size = 20
        self.center_weight = 2.0
        self.size_weight = 1.0
        self.conf_weight = 1.5

    def preprocess_frame(self, frame):
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(frame, 5, 75, 75)
        
        # Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def calculate_priority(self, detection):
        x, y, w, h = detection[:4]
        conf = detection[4]
        
        # Center distance score
        cx = x + w/2
        cy = y + h/2
        dx = cx - self.width/2
        dy = cy - self.height/2
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Weighted scoring
        dist_score = 1.0 - min(1.0, dist/(self.width/2))
        size_score = min(1.0, (w*h)/(self.width*self.height/4))
        
        return (self.center_weight * dist_score +
                self.size_weight * size_score + 
                self.conf_weight * conf)

    def filter_detections(self, detections):
        valid = []
        for det in detections:
            if det[2] >= self.min_target_size and det[3] >= self.min_target_size:
                priority = self.calculate_priority(det)
                valid.append((priority, det))
                
        valid.sort(reverse=True, key=lambda x: x[0])
        return [det for _, det in valid]
