"""
main.py

The main file contains two thread classes:
- `class MainThread(QtCore.QThread)`
    - Back-end thread
    - Handles camera access, motion tracking
    and counting reps
- `class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow)`
    - Front-end thread
    - Handles user input to the graphical user interface

see "doc/main.md" for more details

"""

import sys
import time
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from statistics import mean
from coral.coral import Coral
from app_util import gui_main, gui_thresh
from app_util.config import Config
from app_util.util import Util
from app_util.motion import Motion, Hand, Faces
from app_util.file import File, CsvFile, VideoFile
from app_util.aruco import Aruco
from app_util.playback import Playback
from app_util.movement import (
    ArmExtensions,
    SitToStand, 
    StepTracker, 
    BoxAndBlocks, 
    BoundaryDetector, 
    StandingTimer,
)


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "12/04/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen", 
    "Live Myklebust", 
    "Amber Spurway",
]


class MainThread(QtCore.QThread, Config):
    """
    back-end worker thread: handles camera access, motion tracking
    and counting reps

    """

    image = QtCore.pyqtSignal(QtGui.QImage)
    frame_rate = QtCore.pyqtSignal(float)
    session_time = QtCore.pyqtSignal(int)
    camera_source = QtCore.pyqtSignal(int)
    refresh_source = QtCore.pyqtSignal(int)

    """ 
    back-end signals to handle counting reps 
    
    """
    right_arm_ext = QtCore.pyqtSignal(str)
    left_arm_ext = QtCore.pyqtSignal(str)
    sit_to_stand = QtCore.pyqtSignal(str)

    right_steps = QtCore.pyqtSignal(str)
    left_steps = QtCore.pyqtSignal(str)
    standing_timer = QtCore.pyqtSignal(int)

    right_hand_count = QtCore.pyqtSignal(str)
    left_hand_count = QtCore.pyqtSignal(str)

    """
    tracking modes

    """
    motion_tracking_mode = 0
    steps_tracking_mode = 1
    hand_tracking_mode = 2

    """
    thresholds ids

    """
    sit_to_stand_hip_angle = 0

    left_arm_reach_elbow_angle = 1
    left_arm_reach_shoulder_angle = 2

    right_arm_reach_elbow_angle = 3
    right_arm_reach_shoulder_angle = 4


    def __init__(self, parent=None):
        """"
        init method: initialises the parent classes

        """
        super().__init__(parent)

    def run(self):
        """
        main worker thread

        """
        print(f"Python Version Info: {sys.version}")

        self._active = True
        self.get_source_channel()
        self.start_video_capture()

        """ frame rate (for debugging) """
        frame_times = {"curr time": 0, "prev time": 0}

        """ init motion capture classes """
        self.set_motion_tracking(self._tpu, init=True)
        self._toggle_motion_tracking_method = False

        """ add and init movements """
        self.add_movements()
        self.reset_all_count()

        """ set image width and height to be emmitted to the main-window thread """
        img_width = 1280 - 128/8
        img_height = 720 - 72/8

        """ while camera / video file is opened """
        while self._cap.isOpened() and self._active:

            ret, self._img = self._cap.read()
            self._is_camera_ready = True

            """ if camera not accessed or end of video """
            if ret == False or self._img is None:
                """
                if error accessing camera or end of video and program is not recording:
                - keep iterating through the main while loop until an image signal is received

                """
                if not self._is_recording:
                    continue

                """ 
                if error accessing camera or end of video and program was recording:
                - stop the recording
                - wait until recording starts again (by playing another video) or
                - wait until user switches back to the webcam
                
                """
                self.start_stop_recording()

                while not self._is_recording and self._source != Util.WEBCAM:
                    pass

                continue

            """ create blank frame if 'hide_video' is True """
            self._blank_frame = np.zeros_like(self._img, dtype="uint8") if self._hide_video else None

            """ get frame dimensions """
            self._shape = self._img.shape
            height, width, _ = self._shape

            if not self._tpu:
                self._motion.crop = {"start": Util.INIT, "end": (width, height)}

            """ display frame rate """
            frame_rate = self.get_frame_rate(frame_times)
            self.frame_rate.emit(frame_rate)

            """ 
            corr_mode: used for timeseries correlation with sensors 
            requires txt file with tmestamps corresponding to each frame
            
            """
            if self._corr_mode and self._file_name is not None and not self._file_read:
                file_name = f'{self._file_name.split(".")[0].replace("videos", "time_stamps")}.txt'
                try:
                    file = open(file_name)
                    self._time_stamps = file.readlines()
                    file.close()
                    self._file_read = True
                except FileNotFoundError as err:
                    pass

            """ get the time since start of session """
            if self._start_time is not None and self._is_recording and not self._is_paused:
                
                if self._save_video:
                    self._video_recording.parse_video_frame(self._img, self._session_time)

                """ corr mode: use time-stamps file is provided """
                if self._corr_mode and len(self._time_stamps) > 0:
                    try:
                        self._session_time = float(self._time_stamps[self._index])
                        self._index += 1
                    except IndexError as err:
                        self._session_time = float(self._time_stamps[self._index - 1])
                        self._time_stamps = list()
                        self._index = 0
                else:
                    self._session_time = time.time() - self._start_time - self._pause_time
                
                self.session_time.emit(int(self._session_time))

            """ detect aruco (in hand tracking mode only) """
            self._detected = list()
            if self.hand_tracking_mode in self._modes:
                self._aruco.find_aruco(self._img, self._detected)

            """ detect lines """
            self._img_lines, self._boundary = self._boundary_detector.detect_boundary(
                self._img, self._detected, self._corr_mode,
            )

            """ track motion and count movements (only when recording) """
            if self._is_recording and not self._is_paused:

                self._pose_landmarks = list()

                if self._playback is not None:
                    self._playback.parse_frame(
                        self._img, self._session_time, self._timestamps, overlay=False,
                    )

                if self._tpu:
                    self._tpu_landmarks = self._coral.get_landmarks(self._img)
                else:
                    self._img, begin, end, cropped, view = self._motion.track_motion(
                        self._img, self._pose_landmarks, self._session_time, self._blur_faces,
                        debug=False, dynamic=True, filter=self._filter,
                    )

                """ find faces """
                if self._blur_faces:
                    self._img = self._faces.find_faces(self._img)

                """ count the number of reps for each movement """
                if self._tpu:
                    pass
                else:
                    self.count_movements(cropped, begin, end, view)

                """ draw stick figure overlay (draw after hand detection in "count_movements()) """
                if self._tpu:
                    self._img = self._coral.display_landmarks(self._img, self._tpu_landmarks)
                else:
                    self._motion.draw(
                        self._blank_frame if self._hide_video and self._blank_frame is not None else self._img, 
                        self._pose_landmarks, begin, end
                    )

                """ parse movement data to file object """
                if self._session_time is not None:
                    self._write_file.parse_movements(
                        self._tracking_movements, self._pose_landmarks, self._session_time,
                        self._img.shape, self._curr_movement, self._flip, corr_mode=self._corr_mode,
                    )

            self.emit_qt_img(
                self._blank_frame if self._hide_video and self._blank_frame  is not None else self._img, 
                (width, height), (img_width, img_height)
            )

            """ maintain max frame rate of ~30fps (mainly for smooth video playback) """
            self._delay = self._delay + 0.01 if frame_rate > 30 else 0
            time.sleep(self._delay)

            """ pause video """
            while not self._is_recording and self._source != Util.WEBCAM:
                pass

            if self._toggle_motion_tracking_method == True:
                self._tpu = not self._tpu
                self._toggle_motion_tracking_method = False

        """ handles program exit """
        cv.destroyAllWindows()
        self._cap.release()

    def stop(self):
        """
        stops the worker thread

        """
        self._active = False
        self.wait()

    def set_motion_tracking(self, tpu, init=False):
        if tpu:
            self._coral = Coral()
        else:
            self._motion = Motion(model_complexity=1)   # mediapipe: pose estimation
        
        self._hand = Hand()         # mediapipe: hand tracking
        self._faces = Faces()       # mediapipe: face mesh
        self._aruco = Aruco()       # opencv: aruco detector

        if not init:
            self._toggle_motion_tracking_method = True

    def emit_qt_img(self, img, size, img_size):

        width, height = size
        img_width, img_height = img_size
        
        """ flip image if accessed from webcam """
        if self._source == Util.WEBCAM and self._flip:
            img = cv.flip(img, 1)

        """ emit image signal to the main-window thread to be displayed """
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(
            img.data, width, height, QtGui.QImage.Format_RGB888
        ).scaled(int(img_width), int(img_height), QtCore.Qt.KeepAspectRatio)
        self.image.emit(QtImg)

    def start_video_capture(self, source=None):
        """
        starts video capture from webcam by default

        """
        if self._source == Util.WEBCAM and source is None:
            return
        
        if source is not None:
            if self._is_camera_ready:
                self._cap = self.get_video_capture(self._cap, source=source)
        else:
            self._cap = self.get_video_capture(self._cap)

    def get_video_capture(self, cap, name=None, source=None):
        """
        get video capture from webcam or video file

        """
        if name is not None:
            self._source = Util.VIDEO
            cap = cv.VideoCapture(name)
            cap = self.set_frame_dimensions(cap, "video")

            if not self._is_recording:
                self.start_stop_recording()
            else:
                self.start_stop_recording()
                self.start_stop_recording()

            self._start_time = time.time()
            self.reset_all_count()

            if cap.isOpened():
                return cap
            
        self._source = Util.WEBCAM

        if source is not None:
            self._curr_video_source = source

        self._flip = True if self._curr_video_source == 0 else False
            
        cap = cv.VideoCapture(self._curr_video_source, cv.CAP_DSHOW)
        cap = self.set_frame_dimensions(cap, "webcam")

        if self._is_recording and source is None:
            self.start_stop_recording()

        if cap.isOpened():
            return cap

        print("error opening video stream or file")
        return None
    
    def get_source_channel(self, update=False):
        """
        gets the available webcam source channels and updates combo-box in main window thread

        """
        if update:
            self.refresh_source.emit(0)

        channel = 0
        while channel < Util.MAX_NUM_CAMERAS:
            test_cap = cv.VideoCapture(channel, cv.CAP_DSHOW)

            if test_cap.read()[0]:
                self.camera_source.emit(channel)
            else:
                break

            print(channel)
            channel += 1

    def set_frame_dimensions(self, cap, source):
        """
        set the camera or video resolution and show in terminal (for debugging)

        """
        if cap is not None:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, Util.FRAME_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, Util.FRAME_HEIGHT)

            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            print(f"{source} resolution: {int(width)} x {int(height)}")

        return cap

    def get_file(self, name):
        """
        get the file specified by the user

        """
        self._file_name = name

        self._read_file = File()
        file_type = self._read_file.get_file_type(name)

        """ check that the file is valid and supported by program """
        if file_type == Util.MP4:
            self._cap = self.get_video_capture(self._cap, name=name)
            self._file_read = False
            print(f'video file: "{name}"')

            self._playback = Playback()
            self._timestamps = self._playback.read_timestamps(
                f'{name.replace(".mp4", ".txt")}',
            )

        elif file_type == Util.AVI:
            self._cap = self.get_video_capture(self._cap, name=name)
            self._file_read = False
            print(f'video file: "{name}"')

            self._playback = Playback()
            self._timestamps = self._playback.read_timestamps(
                f'{name.replace(".avi", ".txt")}',
            )

        elif file_type == Util.CSV:
            print(f'csv file: "{name}"')

        elif file_type == Util.FILE_NOT_SUPPORTED:
            invalid_file_msg_box = QtWidgets.QMessageBox()
            invalid_file_msg_box.setWindowTitle("File Not Supported")
            invalid_file_msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            invalid_file_msg_box.setText(
                "The file you have chosen is not supported by the program.\n\n"
                + "Please choose a different file and try again."
            )
            #invalid_file_msg_box.setWindowIcon(get_icon())
            invalid_file_msg_box.exec()

    def handle_exit(self, event):
        """
        handles user exit
        will prompt the user to save recording if user exits while recording in active

        """

        if self._is_recording and self._save_file:

            handle_exit_msg_box = QtWidgets.QMessageBox()
            handle_exit_msg_box.setWindowTitle("Save session?")
            handle_exit_msg_box.setIcon(QtWidgets.QMessageBox.Question)
            handle_exit_msg_box.setText("Would you like to save the recorded session?")
            handle_exit_msg_box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            #handle_exit_msg_box.setWindowIcon(get_icon())
            handle_exit_msg_box.exec()

            button_handler = handle_exit_msg_box.clickedButton()
            button_clicked = handle_exit_msg_box.standardButton(button_handler)
            if button_clicked == QtWidgets.QMessageBox.Yes:
                self._write_file.write(self._name_id, self._filetime)

    def get_frame_rate(self, frame_times):
        """
        calculates frame rate: used for testing

        """
        frame_times["curr time"] = time.time()

        try:
            frame_rate = 1 / (frame_times["curr time"] - frame_times["prev time"])
        except ZeroDivisionError as err:
            frame_rate = 0

        frame_times["prev time"] = frame_times["curr time"]

        return frame_rate

    def start_stop_recording(self):
        """
        starts and stops recording
        called from the main window thread whenever the start/stop button is pressed

        """
        self._is_recording = not self._is_recording

        if self._is_recording:
            """
            create new file object

            """
            self._filetime = Util.create_filename()

            self._write_file = CsvFile(save=self._save_file)

            if self._save_video:
                self._video_recording = VideoFile(save=self._save_video)
                self._video_recording.start_video(self._filetime, self._img.shape)

            if self._stop_time is not None and (
                self._source == Util.VIDEO or self._is_paused
            ):
                self._start_time = time.time() - (self._stop_time - self._start_time)
            else:
                self._start_time = time.time()

            """ resets all movement count if accessed from webcam """
            if self._source == Util.WEBCAM:
                self.reset_all_count()

        else:
            self._stop_time = time.time()

            """ write to csv file """
            self._write_file.write(self._name_id, self._filetime)

            if self._save_video:
                self._video_recording.end_video()

    def pause(self):
        """
        pauses the recording

        """
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_start_time = time.time()
        else:
            self._pause_stop_time = time.time()
            self._pause_time += self._pause_stop_time - self._pause_start_time

    def update_modes(self, mode, update=None):
        """
        callback for updating (adding or removing) modes
        called from the main window thread

        """
        if update == "add":
            self._modes.add(mode)
        elif update == "rm":
            self._modes.discard(mode)

    def reset_all_count(self):
        """
        resets count for all movements
        make sure to update when adding new movements

        """
        self._pause_time = 0
        self._is_paused = False

        self._right_arm_ext.reset_count()
        self._left_arm_ext.reset_count()
        self._sit_to_stand.reset_count()

        self._left_step_tracker.reset_count()
        self._right_step_tracker.reset_count()
        self._standing_timer.reset_time()

        self._box_and_blocks_left.reset_count()
        self._box_and_blocks_right.reset_count()
        self.right_hand_count.emit(str(0))
        self.left_hand_count.emit(str(0))

    def add_movements(self):
        """
        add step tracking and standing timer

        """
        is_steps_enabled = self.steps_tracking_mode in self._modes

        self._left_step_tracker = StepTracker(is_steps_enabled, Util.LEFT, debug=False)
        self._tracking_movements.update({"left steps": self._left_step_tracker})

        self._right_step_tracker = StepTracker(is_steps_enabled, Util.RIGHT, debug=False)
        self._tracking_movements.update({"right steps": self._right_step_tracker})

        self._standing_timer = StandingTimer(is_steps_enabled, debug=False)

        """ 
        add "box and blocks" counting (hand tracking)

        """
        is_hands_enabled = self.hand_tracking_mode in self._modes
        self._boundary_detector = BoundaryDetector(aruco=True)

        self._box_and_blocks_left = BoxAndBlocks(is_hands_enabled, Util.LEFT, debug=False)
        self._tracking_movements.update({"left hand": self._box_and_blocks_left})

        self._box_and_blocks_right = BoxAndBlocks(is_hands_enabled, Util.RIGHT, debug=False)
        self._tracking_movements.update({"right hand": self._box_and_blocks_right})

        """
        add general motion tracking (arm extenstions & sit to stand)

        """
        is_motion_enabled = self.motion_tracking_mode in self._modes

        self._right_arm_ext = ArmExtensions(is_motion_enabled, Util.RIGHT, debug=False)
        self._tracking_movements.update({"right arm ext": self._right_arm_ext})

        self._left_arm_ext = ArmExtensions(is_motion_enabled, Util.LEFT, debug=False)
        self._tracking_movements.update({"left arm ext": self._left_arm_ext})

        self._sit_to_stand = SitToStand(is_motion_enabled, ignore_vis=True, debug=False)
        self._tracking_movements.update({"sit to stand": self._sit_to_stand})

    def count_movements(self, cropped, begin, end, view):
        """ 
        count movements for 'box and blocks'
        
        """
        is_hands_enabled = self.hand_tracking_mode in self._modes
        self._box_and_blocks_right.set_tracking_status(is_hands_enabled)
        self._box_and_blocks_left.set_tracking_status(is_hands_enabled)

        if cropped and is_hands_enabled:
            self._hand_landmarks = list()
            self._img, handedness = self._hand.track_hands(
                self._img, self._hand_landmarks, begin, end, self._source
            )
            right = self._box_and_blocks_right.track_movement(self._hand_landmarks, self._boundary, handedness)
            left = self._box_and_blocks_left.track_movement(self._hand_landmarks, self._boundary, handedness)
            self.right_hand_count.emit(str(right))
            self.left_hand_count.emit(str(left))

        """
        count movements for steps tracking

        """
        is_steps_enabled = self.steps_tracking_mode in self._modes

        """ track left steps """
        self._left_step_tracker.set_tracking_status(is_steps_enabled)
        if self._left_step_tracker.get_tracking_status():
            self._img, left_count = self._left_step_tracker.track_movement(
                self._pose_landmarks, self._img, self._source, view
            )
            self.left_steps.emit(str(left_count))

        """ track right steps """
        self._right_step_tracker.set_tracking_status(is_steps_enabled)
        if self._right_step_tracker.get_tracking_status():
            self._img, right_count = self._right_step_tracker.track_movement(
                self._pose_landmarks, self._img, self._source, view
            )
            self.right_steps.emit(str(right_count))

        """ track standing time """
        self._standing_timer.set_tracking_status(is_steps_enabled)
        if self._standing_timer.get_tracking_status():
            timer = self._standing_timer.track_movement(self._img, self._pose_landmarks)
            self.standing_timer.emit(timer)

        """
        count movements for arm extensions and sit to stand

        """
        is_motion_enabled = self.motion_tracking_mode in self._modes

        """ right arm extensions """
        self._right_arm_ext.set_tracking_status(is_motion_enabled)
        if self._right_arm_ext.get_tracking_status():
            self._img, self._right_arm_ext_count = self._right_arm_ext.track_movement(
                self._pose_landmarks, self._img, self._source
            )
            self.right_arm_ext.emit(str(self._right_arm_ext_count))

        """ left arm extensions """
        self._left_arm_ext.set_tracking_status(is_motion_enabled)
        if self._left_arm_ext.get_tracking_status():
            self._img, self._left_arm_ext_count = self._left_arm_ext.track_movement(
                self._pose_landmarks, self._img, self._source
            )
            self.left_arm_ext.emit(str(self._left_arm_ext_count))

        """ sit to stand """
        self._sit_to_stand.set_tracking_status(is_motion_enabled)
        if self._sit_to_stand.get_tracking_status():
            self._img, self._sit_to_stand_count = self._sit_to_stand.track_movement(
                self._pose_landmarks, self._img, self._source,
            )
            self.sit_to_stand.emit(str(self._sit_to_stand_count))
    
    def adjust_thresh(self, idx, value):
        if idx == self.left_arm_reach_elbow_angle:
            self._left_arm_ext.left_arm_reach_elbow_angle = value

    def get_tracking_movements(self): 
        return self._tracking_movements.copy()
    
    def get_current_mode(self):
        return self._modes.copy()

    def get_recording_status(self):
        return self._is_recording

    def get_input_source(self):
        return self._source
    
    def get_corr_mode(self):
        return self._corr_mode
    
    def get_pause_status(self):
        return self._is_paused

    def update_name_id(self, name_id):
        self._name_id = name_id

    def update_movement(self, movement):
        self._curr_movement = movement

    def generate_file(self, is_checked):
        self._save_file = is_checked

    def blur_faces(self, is_checked):
        self._blur_faces = is_checked

    def toggle_filter(self, is_checked):
        self._filter = is_checked

    def toggle_video(self, is_checked):
        self._hide_video = is_checked
    

class ThreshWindow(QtWidgets.QMainWindow, QtWidgets.QWidget, gui_thresh.Ui_MainWindow):

    sit_to_stand_hip_angle = QtCore.pyqtSignal(int)

    left_arm_reach_elbow_angle = QtCore.pyqtSignal(int)
    left_arm_reach_shoulder_angle = QtCore.pyqtSignal(int)

    right_arm_reach_elbow_angle = QtCore.pyqtSignal(int)
    right_arm_reach_shoulder_angle = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        """ set up gui """
        self.setupUi(self)
        self.setWindowTitle("PhysiCam - Adjust Thresholds")
        self.setWindowIcon(get_icon())

        self.sitToStand_hipAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.sit_to_stand_hip_angle.emit(int(value))
        )
        self.leftArmReach_elbowAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.left_arm_reach_elbow_angle.emit(int(value))
        )
        self.leftArmReach_shoulderAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.left_arm_reach_shoulder_angle.emit(int(value))
        )
        self.rightArmReach_elbowAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.right_arm_reach_elbow_angle.emit(int(value))
        )
        self.rightArmReach_shoulderAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.right_arm_reach_shoulder_angle.emit(int(value))
        )


class MainWindow(QtWidgets.QMainWindow, QtWidgets.QWidget, gui_main.Ui_MainWindow):
    """
    front-end main-window thread: handles graphical user interface

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        """ track mouse """
        self.setMouseTracking(True)

        """ set up gui """
        self.setupUi(self)
        self.setWindowTitle("PhysiCam")
        self.setWindowIcon(get_icon())

        """ create the worker thread """
        self._main_thread = MainThread()
        self._main_thread.start()

        """ connect back-end signals """
        self._main_thread.image.connect(self.update_frame)
        self._main_thread.frame_rate.connect(self.display_frame_rate)
        self._main_thread.session_time.connect(
            lambda time: self.sessiontime_label.setText(f"Session Time: {self.format_time(time)}")
        )

        """ camera source combo-box signals """
        self._main_thread.camera_source.connect(
            lambda source: self.source_comboBox.addItem(str(source))
        )
        self._main_thread.refresh_source.connect(
            lambda: self.source_comboBox.clear()
        )
        self.source_comboBox.currentIndexChanged.connect(
            lambda text: self._main_thread.start_video_capture(source=int(text))
        )

        """ connect motion traking signals """
        self._main_thread.right_arm_ext.connect(
            lambda count: self.right_arm_ext_count_label.setText(f"Right Arm Reach: {count}")
        )
        self._main_thread.left_arm_ext.connect(
            lambda count: self.left_arm_ext_count_label.setText(f"Left Arm Reach: {count}")
        )
        self._main_thread.sit_to_stand.connect(
            lambda count: self.sit_to_stand_count_label.setText(f"Sit to Stand: {count}")
        )
        self._main_thread.right_steps.connect(
            lambda count: self.right_steps_count_label.setText(f"Right Steps: {count}")
        )
        self._main_thread.left_steps.connect(
            lambda count: self.left_steps_count_label.setText(f"Left Steps: {count}")
        )
        self._main_thread.standing_timer.connect(
            lambda timer: self.standing_time_label.setText(f"Standing Time: {self.format_time(timer)}")
        )
        self._main_thread.left_hand_count.connect(
            lambda count: self.left_hand_count_label.setText(f"Left Hand: {count}")
        )
        self._main_thread.right_hand_count.connect(
            lambda count: self.right_hand_count_label.setText(f"Right Hand: {count}")
        )

        """ connect start/stop pushbutton """
        self.start_pushButton.clicked.connect(self._main_thread.start_stop_recording)
        self.start_pushButton.clicked.connect(self.update_start_pushButton)

        """ connect pause pushbutton """
        self.pause_pushButton.clicked.connect(self._main_thread.pause)

        """ connect line edit """
        self.name_id_lineEdit.editingFinished.connect(
            lambda: self._main_thread.update_name_id(self.name_id_lineEdit.text())
        )

        """ connect movement label signals """
        self.movement_set_pushButton.clicked.connect(self.update_movement)

        """ connect action triggers """
        self.actionOpen.triggered.connect(self.open_file)
        self.actionAdjust_Thresholds.triggered.connect(self.adjust_thresholds)
        self.actionWebcam.triggered.connect(
            lambda: self._main_thread.start_video_capture()
        )
        self.actionGenerate_CSV_File.triggered.connect(
            lambda: self._main_thread.generate_file(self.actionGenerate_CSV_File.isChecked())
        )
        self.actionBlur_Faces.triggered.connect(
            lambda: self._main_thread.blur_faces(self.actionBlur_Faces.isChecked())
        )
        self.actionSmooth_Motion_Tracking.triggered.connect(
            lambda: self._main_thread.toggle_filter(self.actionSmooth_Motion_Tracking.isChecked())
        )
        self.actionHide_Video.triggered.connect(
            lambda: self._main_thread.toggle_video(self.actionHide_Video.isChecked())
        )
        self.actionCoral_TPU.triggered.connect(
            lambda: self._main_thread.set_motion_tracking(self.actionCoral_TPU.isChecked())
        )

        """ connect check-box signals """
        self.motion_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.motion_tracking_checkBox, self._main_thread.motion_tracking_mode,
            )
        )
        self.steps_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.steps_tracking_checkBox, self._main_thread.steps_tracking_mode,
            )
        )
        self.hand_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.hand_tracking_checkBox, self._main_thread.hand_tracking_mode,
            )
        )

        """ enable default tracking mode(s) """
        if self._main_thread.get_corr_mode():
            self.hand_tracking_checkBox.setChecked(True)
        else:
            self.motion_tracking_checkBox.setChecked(True)

        self._frame_rates = []

    def update_frame(self, img):
        """
        updated gui interface whenever a new video frame is received from the
        main worker thread

        """
        self.img_label.setPixmap(QtGui.QPixmap(img))

        if self._main_thread.get_input_source() == Util.VIDEO:
            self.source_comboBox.setEnabled(False)
        else:
            self.source_comboBox.setEnabled(True)

        if self._main_thread.get_recording_status():
            self.start_pushButton.setText("Stop")

            if self._main_thread.get_input_source() == Util.WEBCAM:
                self.pause_pushButton.setVisible(True)

        else:
            self.start_pushButton.setText("Start")
            self.pause_pushButton.setVisible(False)

        if self._main_thread.get_pause_status():
            self.pause_pushButton.setText("Resume")
        else:
            self.pause_pushButton.setText("Pause")

        """ update labels based on current mode """
        modes = self._main_thread.get_current_mode()

        if self._main_thread.motion_tracking_mode not in modes:
            self.motion_tracking_groupBox.setDisabled(True)
        else:
            self.motion_tracking_groupBox.setDisabled(False)

        if self._main_thread.steps_tracking_mode not in modes:
            self.steps_tracking_groupBox.setDisabled(True)
        else:
            self.steps_tracking_groupBox.setDisabled(False)

        if self._main_thread.hand_tracking_mode not in modes:
            self.hand_tracking_groupBox.setDisabled(True)
        else:
            self.hand_tracking_groupBox.setDisabled(False)

    def format_time(self, time):
        return "%d:%02d:%02d" % (time // 3600, time // 60, time % 60)

    def display_frame_rate(self, frame_rate):
        """
        shows the current frame rate on the gui
        takes the average of the last ten frame rates for smoother output

        """
        self._frame_rates.append(frame_rate)
        if len(self._frame_rates) > 10:
            self.framerate_label.setText(
                f"Frame Rate: {round(mean(self._frame_rates), 1)} fps"
            )
            self._frame_rates = []    

    def update_start_pushButton(self):
        """
        updates the gui interface whenever the start / stop button is pressed

        """
        self._movements = self._main_thread.get_tracking_movements()

        if self._main_thread.get_recording_status():
            """
            print tracking movements status to terminal
            (only used for testing)

            """
            print("")
            for name, movement in self._movements.items():
                print(f"{name}: {movement.get_tracking_status()}")
            print("")

    def update_movement(self):
        """
        
        """
        if self.movement_set_pushButton.isChecked():
            movement = self.movement_lineEdit.text()

            if movement != str():
                self._main_thread.update_movement(movement)
                self.movement_lineEdit.setEnabled(False)
            else:
                self.movement_set_pushButton.setChecked(False)

        else:
            self._main_thread.update_movement(str())
            self.movement_lineEdit.setEnabled(True)

    def open_file(self):
        """
        callback for when the open action is triggered from the file menu
        gets the file name / path and sends to the main-worker thread

        """
        self._file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "./")
        self._main_thread.get_file(self._file_name[0])

    def adjust_thresholds(self):
        self._thresh_window = ThreshWindow()
        self._thresh_window.show()

        self._thresh_window.sit_to_stand_hip_angle.connect(
            lambda value: self._main_thread.adjust_thresh(
                self._main_thread.sit_to_stand_hip_angle, value,
            )
        )
        self._thresh_window.left_arm_reach_elbow_angle.connect(
            lambda value: self._main_thread.adjust_thresh(
                self._main_thread.left_arm_reach_elbow_angle, value,
            )
        )
        self._thresh_window.left_arm_reach_shoulder_angle.connect(
            lambda value: self._main_thread.adjust_thresh(
                self._main_thread.left_arm_reach_shoulder_angle, value,
            )
        )
        self._thresh_window.right_arm_reach_elbow_angle.connect(
            lambda value: self._main_thread.adjust_thresh(
                self._main_thread.right_arm_reach_elbow_angle, value,
            )
        )
        self._thresh_window.right_arm_reach_shoulder_angle.connect(
            lambda value: self._main_thread.adjust_thresh(
                self._main_thread.right_arm_reach_shoulder_angle, value,
            )
        )

    """ 
    callback functions for updating tracking modes 
    
    """
    def update_tracking_mode(self, check_box, mode):
        if check_box.isChecked():
            self._main_thread.update_modes(mode, update="add")
        else:
            self._main_thread.update_modes(mode, update="rm")

    def mousePressEvent(self, event):
        print(f"x: {event.pos().x()}, y: {event.pos().y()}")

    def closeEvent(self, event):
        self._main_thread.handle_exit(event)
        

def get_icon():
    return QtGui.QIcon(Util.ICON_FILE_PATH)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
