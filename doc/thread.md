# Camera Thread File

- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 12/04/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

- `class CameraThread(QtCore.QThread, Config)`
    - Back-end thread
    - Handles camera access, motion tracking
    and counting reps
    - Inherits from the PyQt5 QThread library and the Config class for settings configuration

## Camera Thread methods

`def __init__(self, parent=None)`
- Initialises all variables to be used in this thread.

`def __str__(self)`
- Representation method for debugging using print statements

`def run(self)`
- Main camera thread
- Called when `self.start()` is called

`def stop(self)`
- Stops the camera thread

`def check_cameras(self)`
- checks if there are any valid cameras connected to host device
- handles exception if no valid cameras are detected

`def set_motion_tracking(self, tpu, init=False)`
- sets motion tracking method:
- `tpu` -> bool: whether to use coral ai hardware accelerator
    - tpu enabled: runs the movenet model on coral tpu usb accelerator
    - tpu disabled: runs the mediapipe model on cpu

`def flip_frame(self, flip)`
- flip the frame for the current camera instance
- `flip` -> bool: whether to flip the current frame

`def ignore_primary(self, ignore)`
- ignore the motion tracking for the primary camera
- `ignore` -> bool: whether to ignore current camera thread if primary
(enable to improve performance of secondary camera(s) is primary camera is not used)

`def update_primary(self, status)`
- updates the status of the current camera thread
- `status` -> bool: whether current camera thread is primary or not

`def emit_qt_img(self, img, size, img_size)`
- function to emit the viewfinder image to the main window thread
- `img` -> np.array: image representing the current frame
- `size` -> tuple[int, int]: shape of the current frame 
- `img_size` -> tuple[int, int]: shape of the output frame to be emitted to the window thread

`def start_video_capture(self, source=None)`
- Starts video capture
- Uses webcam by default

`def get_video_capture(self, cap, name=None, source=None)`
- Gets video capture from webcam or video file
- `cap`: video capture object returned from calling `cv2.VideoCapture()`
- `name`: filename of the file to be opened. Defaults to `none` to open webcam.
- returns a `cap` object from `cv2.VideoCapture()`, else returns `None` if failed to create a `cap` object.

`def get_source_channel(self, update=False)`
- gets the available webcam source channels and updates combo-box in main window thread

`def set_frame_dimensions(self, cap, source)`
- Set the camera or video resolution and show in terminal
- `cap`: video capture object
- `source`: video source: video or webcam

`def get_file(self, name)`
- Gets the file specified by the user
- Checks if the file is a valid format supported by the program. See the "File Module" for a list of supported file formats.
- `name`: name of the file to be retrieved

`def handle_exit(self, event)`
- Handles user exit
- Will prompt the user to save recording if user exits while recording in active
- `event`: not currently used

`def get_frame_rate(self, frame_times)`
- Calculates frame rate
- `frame_times`: dictionary containing the timestamps for the previous and current frames.
- Returns the calculate frame rate.

`def start_stop_recording(self)`
- Starts and stops recording
- Called from the main window thread whenever the start/stop button is pressed

`def pause(self)`
- Pauses the recording

`def update_modes(self, mode, update=None)`
- callback for updating (adding or removing) modes called from the main window thread
- `mode`: current mode to add or remove from the modes set
- `update`: "add" (add mode to set) or "rm" (remove mode from set)

`def reset_all_count(self)`
- Resets count for all movements
- Can be used to reset other parameters at the start of a recording session
- Make sure to update when adding new movements

`def decrement_count(self, movement)`
- method for decrementing counts for a specific `movement`
- used only when multiple cameras are used
- will decrement count if the same movement is counted by multiple cameras

`def add_movements(self)`
- Add movements to be tracked
- Calls the "Movement Module"

`def count_movements(self)`
- Used to count movements during a session
- Will only count movements if enabled
- Emits the updated movement count to the main-window thread to be displayed on the user interface.

`def adjust_thresh(self, idx, value, thresh_win)`
- callback function for adjusting angular thresholds