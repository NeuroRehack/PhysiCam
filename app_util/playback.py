"""
playback.py

Handles real-time video playback of videos recorded from the PhysiCam App
Contains methods to handle video timing to ensure real-time playback

Supported files:
- `.csv`
- `.mp4`

see "doc/playback.md" for more details

"""

from time import sleep


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "10/09/2023"
__status__ = "Prototype"
__credits__ = []


class Playback():
    """
    video playback class
    to be used for videos recorded through the PhysiCam program
    video must be in *.avi format with a corresponding timestamps txt file

    """
    def __init__(self):
        """
        init method

        """
        self._timestamps = None
        self._index = 1

    def read_timestamps(self, fname):
        """
        attempt to read from the timestamps file

        """
        try:
            file = open(fname)
            self._timestamps = file.readlines()
            file.close()

            print(f"timestamps filename: {fname}")
            return True
        
        except FileNotFoundError as err:
            return False

    def parse_frame(self, img, session_time, timestamps, overlay=True):

        """ timestamps file does not exist """
        if self._timestamps is None:
            return

        """ end of timestamps file or timestamps file read error """
        if self._index >= len(self._timestamps) or not timestamps:
            return

        """ attempt to read timestamp from file """
        try:
            timestamp = float(self._timestamps[self._index])
        except ValueError as err:
            print(f"Playback.parse_frame: {err} at {self._index = }")
            return

        """ 
        delay until the time-elapsed equals the corresponding timestamp 
        ensures real-time playback

        """
        if timestamp - session_time > 0:
            sleep(timestamp - session_time)

        self._index += 1