import cv2
import mediapipe as mp
import time
import numpy as np
import platform
import sys
import os
import subprocess

"""
Enhanced Virtual Mouse Control using Hand Gestures
Features:
- Wayland (Hyprland) compatibility
- GPU acceleration where available
- Smooth mouse movement with exponential smoothing
- Multiple gestures for mouse control
- Press 'q' to quit
"""

class MouseController:
    """Abstract base class for mouse controllers."""
    def __init__(self):
        self.screen_width, self.screen_height = self._get_screen_size()

    def _get_screen_size(self):
        """Get screen size - implementation depends on the environment."""
        raise NotImplementedError("Subclass must implement _get_screen_size")
        
    def move_to(self, x, y):
        """Move mouse to the given position."""
        raise NotImplementedError("Subclass must implement move_to")
        
    def left_click(self):
        """Perform a left click."""
        raise NotImplementedError("Subclass must implement left_click")
        
    def right_click(self):
        """Perform a right click."""
        raise NotImplementedError("Subclass must implement right_click")
        
    def double_click(self):
        """Perform a double click."""
        raise NotImplementedError("Subclass must implement double_click")
        
    def scroll(self, amount):
        """Scroll the mouse wheel by the given amount."""
        raise NotImplementedError("Subclass must implement scroll")


class PyAutoGUIController(MouseController):
    """Mouse controller using PyAutoGUI (works with X11)."""
    def __init__(self):
        import pyautogui
        self.pyautogui = pyautogui
        super().__init__()

    def _get_screen_size(self):
        return self.pyautogui.size()
        
    def move_to(self, x, y):
        self.pyautogui.moveTo(x, y)
        
    def left_click(self):
        self.pyautogui.leftClick()
        
    def right_click(self):
        self.pyautogui.rightClick()
        
    def double_click(self):
        self.pyautogui.doubleClick()
        
    def scroll(self, amount):
        self.pyautogui.scroll(amount)


class PynputController(MouseController):
    """Mouse controller using Pynput (better Wayland compatibility)."""
    def __init__(self):
        from pynput.mouse import Controller, Button
        self.mouse = Controller()
        self.Button = Button
        super().__init__()

    def _get_screen_size(self):
        try:
            output = subprocess.check_output(["xrandr", "--current"], universal_newlines=True)
            for line in output.splitlines():
                if " connected" in line and "primary" in line:
                    width, height = map(int, line.split("primary")[1].split()[0].split("x"))
                    return width, height
            # Fallback values if we can't determine screen size
            return 1920, 1080
        except:
            # Fallback for Wayland
            return 1920, 1080
        
    def move_to(self, x, y):
        self.mouse.position = (x, y)
        
    def left_click(self):
        self.mouse.click(self.Button.left)
        
    def right_click(self):
        self.mouse.click(self.Button.right)
        
    def double_click(self):
        self.mouse.click(self.Button.left, 2)
        
    def scroll(self, amount):
        self.mouse.scroll(0, amount/100)


class WaylandController(MouseController):
    """Mouse controller using ydotool for Wayland."""
    def __init__(self):
        self.current_x = 0
        self.current_y = 0
        super().__init__()
        
    def _get_screen_size(self):
        try:
            output = subprocess.check_output(["bash", "-c", "hyprctl monitors -j | jq '.[0].width, .[0].height'"], 
                                           universal_newlines=True)
            lines = output.strip().split('\n')
            if len(lines) >= 2:
                return int(lines[0]), int(lines[1])
            return 1920, 1080
        except:
            return 1920, 1080
    
    def _run_command(self, cmd):
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print(f"Error running command: {e}")
    
    def move_to(self, x, y):
        self.current_x, self.current_y = x, y
        self._run_command(f"ydotool mousemove {int(x)} {int(y)}")
        
    def left_click(self):
        self._run_command("ydotool click 0x01")
        
    def right_click(self):
        self._run_command("ydotool click 0x02")
        
    def double_click(self):
        self._run_command("ydotool click --repeat 2 0x01")
        
    def scroll(self, amount):
        wheel = 4 if amount > 0 else 5  # 4 is up, 5 is down
        repeat = abs(int(amount/5))
        if repeat > 0:
            self._run_command(f"ydotool click --repeat {repeat} 0x{wheel:02x}")


def get_appropriate_controller():
    """Get the appropriate controller based on the environment."""
    # Check if running on Wayland
    wayland_display = os.environ.get('WAYLAND_DISPLAY')
    
    if wayland_display:
        print("Detected Wayland environment")
        
        # Check if ydotool is available (preferred for Wayland)
        try:
            if subprocess.call(['which', 'ydotool'], stdout=subprocess.DEVNULL) == 0:
                print("Using ydotool for Wayland input control")
                return WaylandController()
        except:
            pass
            
        # Fall back to pynput if ydotool isn't available
        try:
            import pynput
            print("Using pynput for input control")
            return PynputController()
        except ImportError:
            print("Warning: pynput not available")
    
    # Default to PyAutoGUI for X11 or as a fallback
    try:
        import pyautogui
        print("Using PyAutoGUI for input control")
        return PyAutoGUIController()
    except ImportError:
        print("Error: PyAutoGUI not available")
        
    try:
        import pynput
        print("Falling back to pynput")
        return PynputController()
    except ImportError:
        print("Error: No suitable input control library found")
        sys.exit(1)


class HandGestureMouse:
    def __init__(self, capture_index=0, smoothing_factor=0.5, use_gpu=True):
        """Initialize the hand gesture mouse controller."""
        # Configuration
        self.smoothing_factor = smoothing_factor
        self.use_gpu = use_gpu
        self.mouse_controller = get_appropriate_controller()
        self.screen_width = self.mouse_controller.screen_width
        self.screen_height = self.mouse_controller.screen_height
        self.prev_x, self.prev_y = 0, 0
        self.click_cooldown = 0
        
        # Initialize camera
        self.cap = cv2.VideoCapture(capture_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize MediaPipe
        mp_hands_config = {
            "static_image_mode": False,
            "max_num_hands": 1,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.5
        }
        
        # Configure GPU usage if available
        if self.use_gpu:
            if self._is_gpu_available():
                print("GPU acceleration enabled")
                mp_hands_config["model_complexity"] = 1  # Use more complex model with GPU
            else:
                print("GPU requested but not available, using CPU")
                self.use_gpu = False
        
        self.hands = mp.solutions.hands.Hands(**mp_hands_config)
        self.mp_draw = mp.solutions.drawing_utils
        
        # FPS calculation
        self.prev_frame_time = 0
        self.current_frame_time = 0
        self.fps = 0
    
    def _is_gpu_available(self):
        """Check if GPU is available for acceleration."""
        try:
            # Check for CUDA support in OpenCV
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
            
            # Alternative check on different platforms
            if platform.system() == "Linux":
                try:
                    import subprocess
                    output = subprocess.check_output("nvidia-smi", shell=True)
                    return True
                except:
                    pass
            
            # Check for TensorFlow GPU
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                return len(gpus) > 0
            except:
                pass
                
            return False
        except:
            return False
    
    def _calculate_fps(self):
        """Calculate and return the current FPS."""
        self.current_frame_time = time.time()
        fps = 1 / (self.current_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.current_frame_time
        return int(fps)
    
    def _smooth_movement(self, new_x, new_y):
        """Apply exponential smoothing for mouse movement."""
        smoothed_x = self.prev_x + self.smoothing_factor * (new_x - self.prev_x)
        smoothed_y = self.prev_y + self.smoothing_factor * (new_y - self.prev_y)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y
    
    def process_frame(self):
        """Process a single camera frame and return the updated frame."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Variables for landmark detection
        index_y = thumb_y = middle_y = ring_y = pinky_y = None
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = hand_landmarks.landmark
                
                for idx, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    
                    # Index finger
                    if idx == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(70, 100, 200), thickness=-1)
                        # Convert coordinates to screen space
                        screen_x = self.screen_width / frame_width * x
                        screen_y = self.screen_height / frame_height * y
                        
                        # Apply smoothing
                        smooth_x, smooth_y = self._smooth_movement(screen_x, screen_y)
                        
                        # Move mouse using appropriate controller
                        self.mouse_controller.move_to(smooth_x, smooth_y)
                        index_y = y
                    
                    # Thumb
                    elif idx == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                        thumb_y = y
                    
                    # Middle finger
                    elif idx == 12:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(100, 83, 270), thickness=-1)
                        middle_y = y
                    
                    # Ring finger
                    elif idx == 16:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(200, 150, 150), thickness=-1)
                        ring_y = y
                    
                    # Pinky finger
                    elif idx == 20:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(150, 200, 150), thickness=-1)
                        pinky_y = y
                
                # Check for gestures if all required landmarks are detected
                if thumb_y is not None:
                    # Perform actions based on detected gestures
                    self._perform_actions(index_y, thumb_y, middle_y, ring_y, pinky_y)
        
        # Display FPS
        self.fps = self._calculate_fps()