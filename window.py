"""
window.py

see "doc/window.md" for more details

"""

import time
import cv2 as cv
from PyQt5 import QtWidgets, QtGui
from statistics import mean
from app_util import gui_main
from app_util.util import Util
from thread import CameraThread
from thresh import ThreshWindow


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "19/12/2023"
__status__ = "Prototype"
__credits__ = ["Agnethe Kaasen", "Live Myklebust", "Amber Spurway"]


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
        self.setWindowIcon(Util.get_icon())

        """ num number of cameras """
        self._max_num_cameras = 5
        self._num_channels = 0

        """ init main thread """
        self._main_thread = CameraThread(cam_id=0, primary=True)
        self._main_thread.start()

        """ connect back-end signals """
        self._main_thread.image.connect(self.update_frame)
        self._main_thread.frame_rate.connect(self.display_frame_rate)
        self._main_thread.session_time.connect(
            lambda time: self.sessiontime_label.setText(f"Session Time: {self.format_time(time)}")
        )
        self._main_thread.tpu_error.connect(lambda: self.actionCoral_TPU.setChecked(False))

        """ set up multiple cameras """
        self._main_thread.multiple_cams.connect(self.multiple_cams)

        """ camera source combo-box signals """
        self._main_thread.camera_source.connect(
            lambda source: self.source_comboBox.addItem(str(source))
        )
        self._main_thread.refresh_source.connect(
            lambda: self.source_comboBox.clear()
        )
        self._curr_primary = 0
        self._worker_threads = None
        self.source_comboBox.currentIndexChanged.connect(
            #lambda text: self._main_thread.start_video_capture(source=int(text))
            lambda text: self.primary_source_changed(int(text))
        )

        self.actionOpen.triggered.connect(self.open_file)
        self.actionAdjust_Thresholds.triggered.connect(self.adjust_thresholds)

        """ worker thread counts """
        self._thread_counts = dict()
        self._prev_move_time = 0
        self._prev_counts = {
            Util.RIGHT_ARM_REACH: 0,
            Util.LEFT_ARM_REACH: 0,
            Util.SIT_TO_STAND: 0,
            Util.RIGHT_STEPS: 0,
            Util.LEFT_STEPS: 0,
        }
        self._prev_count_times = self._prev_counts.copy()

        self._frame_rates = list()
        self.start_pushButton.clicked.connect(self.update_start_pushButton)

        self.connect_signals(self._main_thread)
        self.actionCoral_TPU.setEnabled(False)

        """ connect motion traking signals """
        self._main_thread.right_arm_ext.connect(
            lambda count: self.update_count(self._main_thread, Util.RIGHT_ARM_REACH, count)
        )
        self._main_thread.left_arm_ext.connect(
            lambda count: self.update_count(self._main_thread, Util.LEFT_ARM_REACH, count)
        )
        self._main_thread.sit_to_stand.connect(
            lambda count: self.update_count(self._main_thread, Util.SIT_TO_STAND, count)
        )
        self._main_thread.standing_timer.connect(
            lambda timer: self.standing_time_label.setText(f"Standing Time: {self.format_time(timer)}")
        )
        self._main_thread.right_steps.connect(
            lambda count: self.right_steps_count_label.setText(f"{Util.RIGHT_STEPS}: {count}")
        )
        self._main_thread.left_steps.connect(
            lambda count: self.left_steps_count_label.setText(f"{Util.LEFT_STEPS}: {count}")
        )
        self._main_thread.right_hand_count.connect(
            lambda count: self.right_hand_count_label.setText(f"{Util.RIGHT_HAND}: {count}")
        )
        self._main_thread.left_hand_count.connect(
            lambda count: self.left_hand_count_label.setText(f"{Util.LEFT_HAND}: {count}")
        )

    def connect_signals(self, thread):

        """ connect start/stop pushbutton """
        self.start_pushButton.clicked.connect(thread.start_stop_recording)

        """ connect pause pushbutton """
        self.pause_pushButton.clicked.connect(thread.pause)

        """ connect line edit """
        self.name_id_lineEdit.editingFinished.connect(
            lambda: thread.update_name_id(self.name_id_lineEdit.text())
        )

        """ connect movement label signals """
        self.movement_set_pushButton.clicked.connect(self.update_movement)

        """ connect action triggers """
        self.actionWebcam.triggered.connect(
            lambda: thread.start_video_capture()
        )
        self.actionGenerate_CSV_File.triggered.connect(
            lambda: thread.generate_csv_file(self.actionGenerate_CSV_File.isChecked())
        )
        self.actionSave_Video_File.triggered.connect(
            lambda: thread.save_video_file(self.actionGenerate_CSV_File.isChecked())
        )
        self.actionBlur_Faces.triggered.connect(
            lambda: thread.blur_faces(self.actionBlur_Faces.isChecked())
        )
        self.actionSmooth_Motion_Tracking.triggered.connect(
            lambda: thread.toggle_filter(self.actionSmooth_Motion_Tracking.isChecked())
        )
        self.actionHide_Video.triggered.connect(
            lambda: thread.toggle_video(self.actionHide_Video.isChecked())
        )
        self.actionCoral_TPU.triggered.connect(
            lambda: thread.set_motion_tracking(self.actionCoral_TPU.isChecked())
        )
        self.actionFlip_Frame.triggered.connect(self.flip_frame)

        """ connect check-box signals """
        self.motion_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.motion_tracking_checkBox, thread.motion_tracking_mode,
            )
        )
        self.steps_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.steps_tracking_checkBox, thread.steps_tracking_mode,
            )
        )
        self.hand_tracking_checkBox.stateChanged.connect(
            lambda: self.update_tracking_mode(
                self.hand_tracking_checkBox, thread.hand_tracking_mode,
            )
        )

        """ init movement counts for all threads """
        self._thread_counts.update(
            {
                thread: {
                    Util.LEFT_ARM_REACH: 0,
                    Util.RIGHT_ARM_REACH: 0,
                    Util.SIT_TO_STAND: 0,
                    Util.LEFT_STEPS: 0,
                    Util.RIGHT_STEPS: 0,
                    Util.STANDING_TIME: 0,
                }
            }
        )

        """ enable default tracking mode(s) """
        if thread.get_corr_mode():
            self.hand_tracking_checkBox.setChecked(True)
        else:
            self.motion_tracking_checkBox.setChecked(True)

    def flip_frame(self):
        """
        manually flip frame if using external cameras

        """
        self._main_thread.flip_frame(self.actionFlip_Frame.isChecked())

        time.sleep(1)
        for i in range(1, self._num_channels):
            self._worker_threads[i-1].flip_frame(self.actionFlip_Frame.isChecked())

    def update_frame(self, img):
        """
        updated gui interface whenever a new video frame is received from the
        main worker thread

        """
        self.img_label.setPixmap(QtGui.QPixmap(img))

        #if self._main_thread.get_input_source() == Util.VIDEO or self._num_channels > 1:
            #self.source_comboBox.setEnabled(False)
        #else:
            #self.source_comboBox.setEnabled(True)

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
        """
        return session time is specified format

        """
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

    def primary_source_changed(self, new_idx):
        if self._worker_threads is None:
            return
        
        if self._curr_primary == 0:
            self._main_thread.image.disconnect()
            self._main_thread.update_primary(False)
            self._worker_threads[new_idx - 1].update_primary(True)
            self._worker_threads[new_idx - 1].image.connect(self.update_frame)
        elif new_idx == 0:
            self._worker_threads[self._curr_primary - 1].image.disconnect()
            self._worker_threads[self._curr_primary - 1].update_primary(False)
            self._main_thread.update_primary(True)
            self._main_thread.image.connect(self.update_frame)
        else:
            self._worker_threads[self._curr_primary - 1].image.disconnect()
            self._worker_threads[self._curr_primary - 1].update_primary(False)
            self._worker_threads[new_idx - 1].update_primary(True)
            self._worker_threads[new_idx - 1].image.connect(self.update_frame)
        
        self._curr_primary = new_idx

    def multiple_cams(self, channels):
        """
        set up worker threads for multiple cameras
        a new thread is created for every valid camera detected

        """
        if channels == -1:
            self.actionWebcam.setEnabled(False)

        print(f"{channels} cameras found")
        self._num_channels = channels

        time.sleep(0.5)
        self._worker_threads = tuple()
        for i in range(1, self._num_channels):
            if i >= self._max_num_cameras: break

            self._worker_threads += (CameraThread(cam_id=i, primary=False), )
            self._worker_threads[i-1].start()

            self.connect_signals(self._worker_threads[i-1])

            self._worker_threads[i-1].right_arm_ext.connect(
                lambda count: self.update_count(self._worker_threads[i-1], Util.RIGHT_ARM_REACH, count)
            )
            self._worker_threads[i-1].left_arm_ext.connect(
                lambda count: self.update_count(self._worker_threads[i-1], Util.LEFT_ARM_REACH, count)
            )
            self._worker_threads[i-1].sit_to_stand.connect(
                lambda count: self.update_count(self._worker_threads[i-1], Util.SIT_TO_STAND, count)
            )

            self._worker_threads[i-1].right_steps.connect(
                lambda count: self.update_count(self._worker_threads[i-1], Util.RIGHT_STEPS, count)
            )
            self._worker_threads[i-1].left_steps.connect(
                lambda count: self.update_count(self._worker_threads[i-1], Util.LEFT_STEPS, count)
            )

    def update_count(self, thread, movement, count):
        """
        update the movement count values
        ignores any double-ups in movement detections when using multiple cameras
        
        """
        self._thread_counts[thread].update({movement: count})
        new_count = sum(c[movement] for c in list(self._thread_counts.values()))

        if new_count > self._prev_counts[movement]:

            if time.time() > self._prev_count_times[movement] + Util.MULTI_CAM_DELAY_THRESH:
                self._prev_count_times[movement] = time.time()
                self._prev_counts[movement] = new_count
                self.display_count(movement, new_count)
            else:
                thread.decrement_count(movement)
                self._prev_counts[movement] = new_count - 1

    def display_count(self, movement, count):
        """
        display the updated count values to the gui
        
        """
        if movement == Util.SIT_TO_STAND:
            self.sit_to_stand_count_label.setText(f"{movement}: {count}")
        elif movement == Util.RIGHT_ARM_REACH:
            self.right_arm_ext_count_label.setText(f"{movement}: {count}")
        elif movement == Util.LEFT_ARM_REACH:
            self.left_arm_ext_count_label.setText(f"{movement}: {count}")

        elif movement == Util.RIGHT_STEPS:
            self.right_steps_count_label.setText(f"{movement}: {count}")
        elif movement == Util.LEFT_STEPS:
            self.left_steps_count_label.setText(f"{movement}: {count}")

    def update_start_pushButton(self):
        """
        updates the gui interface whenever the start / stop button is pressed

        """
        self._movements = self._main_thread.get_tracking_movements()

        if not self._main_thread.get_recording_status():
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
        updates current movement
        takes in user input from the movement line edit
        outputs the movement label to the generated csv file
        
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
        """
        method to set up the threshold adjustments ui
        (in development)

        """
        try: 
            if time.time() < self._thresh_prev_launch_time + 1:
                del self._thresh_prev_launch_time
                return
        except: 
            self._thresh_prev_launch_time = time.time()

        self._thresh_window = ThreshWindow(self._main_thread.get_tracking_movements())
        self._thresh_window.show()

        self._main_thread.adjust_thresh(self._thresh_window)
        for th in self._worker_threads:
            th.adjust_thresh(self._thresh_window)

    def update_tracking_mode(self, check_box, mode):
        """ 
        callback functions for updating tracking modes 
        
        """
        if check_box.isChecked():
            self._main_thread.update_modes(mode, update="add")
            for i in range(1, self._num_channels):
                if i >= self._max_num_cameras: break
                self._worker_threads[i-1].update_modes(mode, update="add")
        else:
            self._main_thread.update_modes(mode, update="rm")
            for i in range(1, self._num_channels):
                if i >= self._max_num_cameras: break
                self._worker_threads[i-1].update_modes(mode, update="rm")

    def mousePressEvent(self, event):
        """
        prints mouse click co-ords to the terminal
        (no used)
        
        """
        print(f"x: {event.pos().x()}, y: {event.pos().y()}")

    def closeEvent(self, event):
        """
        handles user exiting the program
        
        """
        self._main_thread.handle_exit(event)