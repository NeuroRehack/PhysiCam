"""
config.py

see "doc/config.md" for more details

"""

from .util import Util
from .motion import Motion


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "18/05/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen", 
    "Live Myklebust", 
    "Amber Spurway",
]


class Config():
    """
    cofiguration class: inherited by the main thread class

    """
    def __init__(self):

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

        self._read_file = None
        self._write_file = None
        self._save_file = False      ### enable / disable "save to csv"
        self._save_video = False    ### enable / disable saving video recordings
        self._video_recording = None

        self._filetime = Util.create_filename()
        self._file_name = None
        self._file_read = False
        self._time_stamps = list()
        self._index = 0

        self._hide_video = False
        self._filter = True         # lpf to improve motion tracking smoothness
        self._blur_faces = False         
        self._flip = False          ### flip video, TO-DO: save flip status to csv file
        self._name_id = str()
        self._curr_movement = str()

        self._modes = set()

        self._tpu = False       ### Coral TPU enable/disable
        self._ignore = False    ### ignore primary camera

        self._playback = None
        self._timestamps = None

        self._corr_mode = False      ### corr mode (use with motion sensors)

        self._save_file = True if self._corr_mode else self._save_file