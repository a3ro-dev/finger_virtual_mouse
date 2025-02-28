# Finger Virtual Mouse

Control your mouse using hand gestures captured by your webcam. This application uses MediaPipe for hand tracking and supports both X11 and Wayland (including Hyprland).

## Features

- **X11 and Wayland (Hyprland) support**
- **GPU acceleration** where available
- **Smooth mouse movement** with exponential smoothing
- **Multiple gestures** for different mouse actions:
  - INDEX FINGER + THUMB = LEFT CLICK
  - MIDDLE FINGER + THUMB = RIGHT CLICK
  - RING FINGER + THUMB = DOUBLE CLICK
  - PINKY + THUMB = SCROLL MODE
- **Real-time FPS display** to monitor performance

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- PyAutoGUI (for X11) or pynput/ydotool (for Wayland)
- NumPy

## Installation

1. Clone the repository
```bash
git clone https://github.com/a3ro-dev/finger_virtual_mouse
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Run the program
```bash
python mouse.py
```

## Contributing
Any contributions are appreciated. Feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

