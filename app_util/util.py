"""
util.py

Contains static definitions specifically used for this project.

see "doc/util.md" for more details

"""
import time
import numpy as np
from scipy import signal as sig


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "22/04/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen", 
    "Live Myklebust", 
    "Amber Spurway"
]


class Util:

    """ default path for csv files to be saved """
    DEFAULT_FILE_PATH = "./PhysiCam_RecordedSessions"
    DEFAULT_VIDEO_PATH = "./Physicam_VideoRecordings"

    """ icon file path """
    ICON_FILE_PATH = "./app_util/images/icon.png"

    """ supported files """
    FILE_NOT_SUPPORTED = -1
    CSV = 0
    MP4 = 1
    AVI = 2

    """ max frame dimensions (full-hd): 1920 x 1080 """
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    FRAME_ORIGIN = 0

    """ positional min and max thresholds """
    MIN = 0.01
    MAX = 0.99

    """ visibility threshold """
    VIS = 50

    """ input source definitions """
    VIDEO = 0
    WEBCAM = 1

    MAX_NUM_CAMERAS = 64

    """ pre-defined colours (b, g, r) """
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    BORDER_WIDTH = 3

    """ camera angle definitions """
    SIDE_VIEW = 0
    FRONT_VIEW = 1
    REAR_VIEW = 2

    """ init tuple (used for bounding box) """
    INIT = (0, 0)

    """ co-ordinate definitions """
    X = 0
    Y = 1

    """ left or right """
    LEFT = "Left"
    RIGHT = "Right"

    """ movement labels """
    RIGHT_ARM_REACH = "Right Arm Reach"
    LEFT_ARM_REACH = "Left Arm Reach"
    SIT_TO_STAND = "Sit to Stand"
    RIGHT_STEPS = "Right Steps"
    LEFT_STEPS = "Left Steps"
    STANDING_TIME = "Standing Time"
    LEFT_HAND = "Left Hand"
    RIGHT_HAND = "Right Hand"


    def get_points(array, indices):
        out = list()
        for i in indices:
            out.append(array[i])

        return out.copy()
    
    def create_filename():
        """
        creates a unique filename using the current system time and date

        """
        return f'{time.strftime("%y%m%d-%H%M%S")}'
    
    def low_pass_filter(data, fc, fs, order):
        """
        butterworth low pass filter

        """
        w = fc / (fs / 2)
        sos = sig.butter(order, w, btype="low", analog=False, output="sos")
        out = sig.sosfilt(sos, np.array(data))

        return out.copy()
