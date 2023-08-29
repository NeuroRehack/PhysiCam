import sys
import time
import cv2 as cv
from PyQt5 import QtCore, QtWidgets, QtGui
from statistics import mean

from ml_gui import Ui_MainWindow

sys.path.append("../")
from app_util.util import Util
from app_util.motion import Motion, Hand
from app_util.file import File, CsvFile
from app_util.aruco import Aruco


class MainThread(QtCore.QThread):
    """
    back-end worker thread: handles camera access, motion tracking
    and counting reps

    """

    image = QtCore.pyqtSignal(QtGui.QImage)
    frame_rate = QtCore.pyqtSignal(float)
    session_time = QtCore.pyqtSignal(int)
    camera_source = QtCore.pyqtSignal(int)
    refresh_source = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._cap = None
        self._is_recording = False
        self._is_paused = False
        self._is_camera_ready = False
        self._pause_time = 0

        self._tracking_movements = dict()
        self._start_time = None
        self._stop_time = None
        self._session_time = None

        self._curr_video_source = 0
        self._source = None
        self._shape = None
        self._delay = 0

    def run(self):
        """
        main worker thread

        """
        self._active = True
        self.get_source_channel()
        self.start_video_capture()

        """ frame rate (for debugging) """
        frame_times = {"curr time": 0, "prev time": 0}

        """ init motion capture """
        self._motion = Motion()

        """ init hand tracker """
        self._hand = Hand()

        """ init aruco detector """
        self._aruco = Aruco()

        """ set image width and height to be emmitted to the main-window thread """
        img_width = 1280 - 128/8
        img_height = 720 - 72/8

        prev_update_time = time.time()

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

            """ get frame dimensions """
            self._shape = self._img.shape
            height, width, _ = self._shape
            self._motion.crop = {"start": Util.INIT, "end": (width, height)}

            """ display frame rate """
            frame_rate = self.get_frame_rate(frame_times)
            self.frame_rate.emit(frame_rate)

            """ get the time since start of session """
            if (
                self._start_time is not None
                and self._is_recording
                and not self._is_paused
            ):
                self._session_time = time.time() - self._start_time - self._pause_time
                self.session_time.emit(int(self._session_time))

            """ track motion and count movements (only when recording) """
            if self._is_recording and not self._is_paused:

                # self._img = self._hand.find_hand(self._img)

                self._pose_landmarks = list()
                self._img, begin, end, cropped, view = self._motion.track_motion(
                    self._img,
                    self._pose_landmarks,
                    self._session_time,
                    debug=False,
                    dynamic=True,
                )

                """ draw stick figure overlay (draw after hand detection in "count_movements()) """
                self._motion.draw(self._img, self._pose_landmarks)

            """ flip image if accessed from webcam """
            if self._source == Util.WEBCAM:
                self._img = cv.flip(self._img, 1)

            """ emit image signal to the main-window thread to be displayed """
            self._img = cv.cvtColor(self._img, cv.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(
                self._img.data, width, height, QtGui.QImage.Format_RGB888
            ).scaled(int(img_width), int(img_height), QtCore.Qt.KeepAspectRatio)
            self.image.emit(QtImg)

            """ maintain max frame rate of ~30fps (mainly for smooth video playback) """
            self._delay = self._delay + 0.01 if frame_rate > 30 else 0
            time.sleep(self._delay)

            """ pause video """
            while not self._is_recording and self._source != Util.WEBCAM:
                pass

        """ handles program exit """
        cv.destroyAllWindows()
        self._cap.release()

    def stop(self):
        """
        stops the worker thread

        """
        self._active = False
        self.wait()

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
            #self.reset_all_count()

            if cap.isOpened():
                return cap
            
        self._source = Util.WEBCAM

        if source is not None:
            self._curr_video_source = source
            
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

    def get_recording_status(self):
        """
        gets current recording status
        returns True if recording, else returns False
        used in the main window thread to update gui

        """
        return self._is_recording

    def get_input_source(self):
        """
        gets current input source (video or webcam)

        """
        return self._source
    
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
            #self._write_file = CsvFile(save=self._save_file)

            if self._stop_time is not None and (
                self._source == Util.VIDEO or self._is_paused
            ):
                self._start_time = time.time() - (self._stop_time - self._start_time)
            else:
                self._start_time = time.time()

            """ resets all movement count if accessed from webcam """
            if self._source == Util.WEBCAM:
                pass
                #self.reset_all_count()

        else:
            self._stop_time = time.time()

            """ write to csv file """
            #self._write_file.write(self._name_id, self._shape)


class MainWindow(QtWidgets.QMainWindow, QtWidgets.QWidget, Ui_MainWindow):
    """
    front-end main-window thread: handles graphical user interface

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        """ track mouse """
        self.setMouseTracking(True)

        """ set up gui """
        self.setupUi(self)
        self.setWindowTitle("PhysiCam ML: Collect Data")

        """ create the worker thread """
        self._main_thread = MainThread()
        self._main_thread.start()

        """ connect back-end signals """
        self._main_thread.image.connect(self.update_frame)
        self._main_thread.frame_rate.connect(self.display_frame_rate)
        """ camera source combo-box signals """
        self._main_thread.camera_source.connect(self.get_source)
        self._main_thread.refresh_source.connect(self.refresh_source)
        self.source_comboBox.currentIndexChanged.connect(self.update_source)

        """ connect start/stop pushbutton """
        self.start_pushButton.clicked.connect(self._main_thread.start_stop_recording)

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

        else:
            self.start_pushButton.setText("Start")

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

    def get_source(self, source):
        """
        callback for adding webcam source channels to combo-box
        used at the start of the program to get all available channels

        """
        self.source_comboBox.addItem(str(source))

    def update_source(self, text):
        """
        callback for updating the current webcam source channel
        used when switching between webcams

        """
        self._main_thread.start_video_capture(source=int(text))

    def refresh_source(self):
        self.source_comboBox.clear()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
