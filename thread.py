"""
thread.py

see "doc/thread.md" for more details

"""

import time
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from app_util.config import Config
from app_util.util import Util
from app_util.motion import Motion, Hand, Faces
from app_util.file import File, CsvFile, VideoFile
from app_util.aruco import Aruco
from app_util.playback import Playback
from app_util.movement import (
    ArmExtensions, SitToStand, StepTracker, BoxAndBlocks, BoundaryDetector, StandingTimer,
)


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "19/12/2023"
__status__ = "Prototype"
__credits__ = ["Agnethe Kaasen", "Live Myklebust", "Amber Spurway"]


class CameraThread(QtCore.QThread, Config):
    """
    back-end worker thread: handles camera access, motion tracking
    and counting reps

    """
    image = QtCore.pyqtSignal(QtGui.QImage)
    frame_rate = QtCore.pyqtSignal(float)
    session_time = QtCore.pyqtSignal(int)
    camera_source = QtCore.pyqtSignal(int)
    refresh_source = QtCore.pyqtSignal(int)
    tpu_error = QtCore.pyqtSignal(int)
    multiple_cams = QtCore.pyqtSignal(int)

    """ back-end signals to handle counting reps """
    right_arm_ext = QtCore.pyqtSignal(int)
    left_arm_ext = QtCore.pyqtSignal(int)
    sit_to_stand = QtCore.pyqtSignal(int)
    right_steps = QtCore.pyqtSignal(int)
    left_steps = QtCore.pyqtSignal(int)
    standing_timer = QtCore.pyqtSignal(int)
    right_hand_count = QtCore.pyqtSignal(int)
    left_hand_count = QtCore.pyqtSignal(int)

    """ tracking modes """
    motion_tracking_mode = 0
    steps_tracking_mode = 1
    hand_tracking_mode = 2

    """ thresholds ids """
    sit_to_stand_hip_angle = 0
    left_arm_reach_elbow_angle = 1
    left_arm_reach_shoulder_angle = 2
    right_arm_reach_elbow_angle = 3
    right_arm_reach_shoulder_angle = 4

    def __init__(self, cam_id=0, primary=True, parent=None):
        """
        init method: initialises the parent classes

        """
        super().__init__(parent)
        self._cam_id = cam_id
        self._primary = primary

        """ add default motion tracking mode to all secondary camera threads """
        if not self._primary:
            self._modes.add(self.motion_tracking_mode)

    def __str__(self):
        """
        print representation method for worker threads

        """
        return f"thread {self._cam_id}"

    def run(self):
        """
        main worker thread

        """
        self._active = True
        if self._primary: self.get_source_channel()

        """ start video capture """
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

        """ handle device with no cameras """
        self.check_cameras()

        """ while camera / video file is opened """
        while (self._no_cameras_detected or self._cap.isOpened()) and self._active:

            if self._no_cameras_detected:
                ret, self._img = True, np.zeros((Util.FRAME_HEIGHT, Util.FRAME_WIDTH, 3), dtype="uint8")
            else:
                ret, self._img = self._cap.read()

            self._is_camera_ready = True

            """ if camera not accessed or end of video """
            if ret == False or self._img is None:
                """
                if error accessing camera or end of video and program is not recording:
                - keep iterating through the main while loop until an image signal is received

                """
                if not self._is_recording: continue

                """ 
                if error accessing camera or end of video and program was recording:
                - stop the recording
                - wait until recording starts again (by playing another video) or
                - wait until user switches back to the webcam
                
                """
                self.start_stop_recording()

                while not self._is_recording and self._source != Util.WEBCAM: pass
                continue

            """ create blank frame if 'hide_video' or 'ignore_promary' is True """
            self._blank_frame = np.zeros_like(self._img, dtype="uint8") if (
                self._hide_video or self._ignore_primary
             ) else None

            """ get frame dimensions """
            self._shape = self._img.shape
            height, width, _ = self._shape

            if not self._tpu: self._motion.crop = {"start": Util.INIT, "end": (width, height)}

            """ display frame rate """
            frame_rate = self.get_frame_rate(frame_times)
            self.frame_rate.emit(frame_rate)

            """ 
            corr_mode: used for timeseries correlation with sensors 
            requires txt file with timestamps corresponding to each frame
            
            """
            if self._corr_mode and self._file_name is not None and not self._file_read:
                file_name = f'{self._file_name.split(".")[0].replace("videos", "time_stamps")}.txt'
                try:
                    file = open(file_name)
                    self._time_stamps = file.readlines()
                    file.close()
                    self._file_read = True
                except FileNotFoundError as err: print(err)

            """ get the time since start of session """
            if self._start_time is not None and self._is_recording and not self._is_paused:
                
                if self._save_video and not self._ignore_primary:
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

                """ run movenet pose estimation if tpu enabled """
                if self._tpu:
                    self._tpu_landmarks = self._coral.get_landmarks(self._img)
                elif not self._ignore_primary:
                    self._img, begin, end, cropped, view = self._motion.track_motion(
                        self._img, self._pose_landmarks, self._session_time, self._blur_faces,
                        debug=False, dynamic=True, filter=self._filter,
                    )

                """ find faces """
                if self._blur_faces: self._img = self._faces.find_faces(self._img)

                """ count the number of reps for each movement """
                if self._tpu: pass
                elif not self._ignore_primary:
                    self.count_movements(cropped, begin, end, view)

                """ draw stick figure overlay (draw after hand detection in "count_movements()) """
                if self._tpu:
                    self._img = self._coral.display_landmarks(self._img, self._tpu_landmarks)
                elif not self._ignore_primary:
                    self._motion.draw(
                        self._blank_frame if (
                            self._hide_video and self._blank_frame is not None 
                        ) else self._img, self._pose_landmarks, begin, end
                    )

                """ parse movement data to file object """
                if self._session_time is not None and not self._ignore_primary:
                    self._write_file.parse_movements(
                        self._tracking_movements, self._pose_landmarks, self._session_time,
                        self._img.shape, self._curr_movement, self._flip, corr_mode=self._corr_mode,
                    )

            """ display a blank frame if 'hide video' is enabled """
            self.emit_qt_img(
                self._blank_frame if (
                    (self._hide_video or self._ignore_primary) and self._blank_frame is not None
                ) else self._img, (width, height), (img_width, img_height)
            )

            """ maintain max frame rate of ~30fps (mainly for smooth video playback) """
            self._delay = self._delay + 0.03 if frame_rate > 30 else 0
            time.sleep(self._delay)

            """ pause video """
            while not self._is_recording and self._source != Util.WEBCAM: pass

            """ toggle motion tracking method (use cpu or tpu) """
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

    def check_cameras(self):
        """
        check is there are any valid cameras connected to host device
        
        """
        try:
            self._cap.isOpened()
            self._no_cameras_detected = False
        except AttributeError as err:
            print(err)
            self._no_cameras_detected = True
            self.multiple_cams.emit(-1)

    def set_motion_tracking(self, tpu, init=False):
        """
        sets motion tracking method:
        - tpu enabled: runs the movenet model on coral tpu usb accelerator
        - tpu disabled: runs the mediapipe model on cpu

        """
        if tpu:
            try: pass #self._coral = Coral()
            except ValueError as err:
                tpu_error_msg_box = QtWidgets.QMessageBox()
                tpu_error_msg_box.setWindowTitle("Coral TPU Error")
                tpu_error_msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                tpu_error_msg_box.setText(
                    "Unable to set up Coral TPU.\n\n"
                    + "Please check connection and try again."
                )
                #invalid_file_msg_box.setWindowIcon(get_icon())
                tpu_error_msg_box.exec()
                self.tpu_error.emit(1)
                return
            except NameError as err:
                tpu_error_msg_box = QtWidgets.QMessageBox()
                tpu_error_msg_box.setWindowTitle("Coral TPU Error")
                tpu_error_msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                tpu_error_msg_box.setText(
                    "Unable to set up Coral TPU.\n\n"
                    + "Coral TPU device is not supported."
                )
                #invalid_file_msg_box.setWindowIcon(get_icon())
                tpu_error_msg_box.exec()
                self.tpu_error.emit(1)
                return
        else:
            self._motion = Motion(model_complexity=1)   # mediapipe: pose estimation
        
        self._hand = Hand()         # mediapipe: hand tracking
        self._faces = Faces()       # mediapipe: face mesh
        self._aruco = Aruco()       # opencv: aruco detector

        if not init: self._toggle_motion_tracking_method = True

    def flip_frame(self, flip):
        """
        flip the frame for the current camera instance

        """
        self._flip = flip

    def ignore_primary(self, ignore):
        """
        ignore the motion tracking for the primary camera
        (enable to improve performance of secondary camera(s) is primary camera is not used)
        
        """
        if self._primary:
            self._ignore_primary = ignore

    def emit_qt_img(self, img, size, img_size):
        """
        function to emit the viewfinder image to the main window thread
        
        """
        width, height = size
        img_width, img_height = img_size
        
        """ flip image if accessed from webcam """
        if self._source == Util.WEBCAM and self._flip: img = cv.flip(img, 1)

        """ show frame from second camera """
        if not self._primary and not self._no_cameras_detected:
            cv.imshow(f"PhysiCam: {self._cam_id}", img)
            cv.waitKey(1)

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
        if self._source == Util.WEBCAM and source is None: return
        
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

            if not self._is_recording: self.start_stop_recording()
            else:
                self.start_stop_recording()
                self.start_stop_recording()

            self._start_time = time.time()
            self.reset_all_count()

            if cap.isOpened(): return cap
            
        self._source = Util.WEBCAM
        if source is not None: self._curr_video_source = source

        self._flip = True if self._curr_video_source == 0 else False
            
        """ start video capture """
        #cap = cv.VideoCapture(self._curr_video_source, cv.CAP_DSHOW)
        cap = cv.VideoCapture(self._cam_id, cv.CAP_DSHOW)
        cap = self.set_frame_dimensions(cap, "webcam")

        if self._is_recording and source is None: self.start_stop_recording()

        if cap.isOpened(): return cap
        print("error opening video stream or file")
        return None
    
    def get_source_channel(self, update=False):
        """
        gets the available webcam source channels and updates combo-box in main window thread

        """
        if update: self.refresh_source.emit(0)

        channel = 0
        while channel < Util.MAX_NUM_CAMERAS:
            test_cap = cv.VideoCapture(channel, cv.CAP_DSHOW)

            if test_cap.read()[0]: self.camera_source.emit(channel)
            else: break
            channel += 1

        if channel > 1: self.multiple_cams.emit(channel)

    def set_frame_dimensions(self, cap, source):
        """
        set the camera or video resolution and show in terminal (for debugging)

        """
        if cap is not None:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, Util.FRAME_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, Util.FRAME_HEIGHT)

            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            print(f"channel {self._cam_id}, {source} resolution: {int(width)} x {int(height)}")

        return cap

    def get_file(self, name):
        """
        get the file specified by the user

        """
        self._file_name = name

        self._read_file = File()
        file_type = self._read_file.get_file_type(name)

        self._no_cameras_detected = False

        """ check that the file is valid and supported by program """
        if file_type == Util.MP4:   # *.mp4 files
            self._cap = self.get_video_capture(self._cap, name=name)
            self._file_read = False
            print(f'video file: "{name}"')

            self._playback = Playback()
            self._timestamps = self._playback.read_timestamps(
                f'{name.replace(".mp4", ".txt")}',
            )

        elif file_type == Util.AVI: # *.avi files
            self._cap = self.get_video_capture(self._cap, name=name)
            self._file_read = False
            print(f'video file: "{name}"')

            self._playback = Playback()
            self._timestamps = self._playback.read_timestamps(
                f'{name.replace(".avi", ".txt")}',
            )

        elif file_type == Util.CSV: # *.csv files
            print(f'csv file: "{name}"')

        elif file_type == Util.FILE_NOT_SUPPORTED:
            """
            launches an invalid file msg box is file type i nt supported

            """
            invalid_file_msg_box = QtWidgets.QMessageBox()
            invalid_file_msg_box.setWindowTitle("File Not Supported")
            invalid_file_msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            invalid_file_msg_box.setText(
                "The file you have chosen is not supported by the program.\n\n"
                + "Please choose a different file and try again."
            )
            #invalid_file_msg_box.setWindowIcon(get_icon())
            invalid_file_msg_box.exec()
            self.check_cameras()

    def handle_exit(self, event):
        """
        handles user exit
        will prompt the user to save recording if user exits while recording in active

        """

        if self._is_recording and self._save_file:
            """
            launches a save file? msg box if user exits program while recording

            """
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
            if button_clicked == QtWidgets.QMessageBox.Yes and not self._ignore_primary:
                self._write_file.write(self._name_id, self._filetime, self._cam_id)

    def get_frame_rate(self, frame_times):
        """
        calculates frame rate: used for testing

        """
        frame_times["curr time"] = time.time()

        try:
            frame_rate = 1 / (frame_times["curr time"] - frame_times["prev time"])
        except ZeroDivisionError as err:
            frame_rate = 0
            #print(err)

        frame_times["prev time"] = frame_times["curr time"]
        return frame_rate

    def start_stop_recording(self):
        """
        starts and stops recording
        called from the main window thread whenever the start/stop button is pressed

        """
        self._is_recording = not self._is_recording
        
        if self._is_recording:
            self._filetime = Util.create_filename() # create new file object
            self._write_file = CsvFile(save=self._save_file)

            if self._save_video:
                self._video_recording = VideoFile(save=self._save_video)
                if not self._ignore_primary:
                    self._video_recording.start_video(self._filetime, self._img.shape, self._cam_id)

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
            if not self._ignore_primary:
                self._write_file.write(self._name_id, self._filetime, self._cam_id)

            if self._save_video and not self._ignore_primary:
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
        if update == "add": self._modes.add(mode)
        elif update == "rm": self._modes.discard(mode)

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
        self.right_hand_count.emit(0)
        self.left_hand_count.emit(0)

    def decrement_count(self, movement):
        """
        method for decrementing counts for a specific 'movement'
        used only when multiple cameras are used
        will decrement count if the same movement is counted by multiple cameras

        """
        if movement == Util.RIGHT_ARM_REACH:
            self._right_arm_ext.decrement_count()
        elif movement == Util.LEFT_ARM_REACH:
            self._left_arm_ext.decrement_count()
        elif movement == Util.SIT_TO_STAND:
            self._sit_to_stand.decrement_count()

    def add_movements(self):
        """
        add step tracking and standing timer

        """
        is_steps_enabled = self.steps_tracking_mode in self._modes
        self._standing_timer = StandingTimer(is_steps_enabled, debug=False)

        self._left_step_tracker = StepTracker(is_steps_enabled, Util.LEFT, debug=False)
        self._tracking_movements.update({"left steps": self._left_step_tracker})

        self._right_step_tracker = StepTracker(is_steps_enabled, Util.RIGHT, debug=False)
        self._tracking_movements.update({"right steps": self._right_step_tracker})

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
            self.right_hand_count.emit(right)
            self.left_hand_count.emit(left)

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
            self.left_steps.emit(left_count)

        """ track right steps """
        self._right_step_tracker.set_tracking_status(is_steps_enabled)
        if self._right_step_tracker.get_tracking_status():
            self._img, right_count = self._right_step_tracker.track_movement(
                self._pose_landmarks, self._img, self._source, view
            )
            self.right_steps.emit(right_count)

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
            self.right_arm_ext.emit(self._right_arm_ext_count)

        """ left arm extensions """
        self._left_arm_ext.set_tracking_status(is_motion_enabled)
        if self._left_arm_ext.get_tracking_status():
            self._img, self._left_arm_ext_count = self._left_arm_ext.track_movement(
                self._pose_landmarks, self._img, self._source
            )
            self.left_arm_ext.emit(self._left_arm_ext_count)

        """ sit to stand """
        self._sit_to_stand.set_tracking_status(is_motion_enabled)
        if self._sit_to_stand.get_tracking_status():
            self._img, self._sit_to_stand_count = self._sit_to_stand.track_movement(
                self._pose_landmarks, self._img, self._source,
            )
            self.sit_to_stand.emit(self._sit_to_stand_count)
    
    def adjust_thresh(self, idx, value, thresh_win):
        """
        calback function for adjusting angular thresholds (in development)

        """
        if idx == self.left_arm_reach_elbow_angle:
            self._left_arm_ext.left_arm_reach_elbow_angle = value
            thresh_win.left_elbow_label.setText(f"{value}")

        elif idx == self.left_arm_reach_shoulder_angle:
            self._left_arm_ext.left_arm_reach_shoulder_angle = value
            thresh_win.left_shoulder_label.setText(f"{value}")

        elif idx == self.right_arm_reach_elbow_angle:
            self._right_arm_ext.right_arm_reach_elbow_angle = value
            thresh_win.right_elbow_label.setText(f"{value}")

        elif idx == self.right_arm_reach_shoulder_angle:
            self._right_arm_ext.right_arm_reach_shoulder_angle = value
            thresh_win.right_shoulder_label.setText(f"{value}")

    """ get functions used in the the main window thread """
    def get_tracking_movements(self): return self._tracking_movements.copy()
    def get_current_mode(self): return self._modes.copy()
    def get_recording_status(self): return self._is_recording
    def get_input_source(self): return self._source
    def get_corr_mode(self): return self._corr_mode
    def get_pause_status(self): return self._is_paused

    """ set/toggle function set by user inputs from the main window thread """
    def update_name_id(self, name_id): self._name_id = name_id
    def update_movement(self, movement): self._curr_movement = movement
    def generate_file(self, is_checked): self._save_file = is_checked
    def blur_faces(self, is_checked): self._blur_faces = is_checked
    def toggle_filter(self, is_checked): self._filter = is_checked
    def toggle_video(self, is_checked): self._hide_video = is_checked