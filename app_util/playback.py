from time import sleep

class Playback():

    def __init__(self):
        self._timestamps = None
        self._index = 1

    def read_timestamps(self, fname):
        try:
            file = open(fname)
            self._timestamps = file.readlines()

            file.close()
            return True
        except FileNotFoundError as err:
            return False

    def parse_frame(self, img, session_time, timestamps, overlay=True):

        if self._timestamps is None:
            print(session_time, 1)
            return

        if self._index >= len(self._timestamps) or not timestamps:
            print(session_time, 2)
            return

        try:
            timestamp = float(self._timestamps[self._index])
        except ValueError as err:
            print(f"Playback.parse_frame: {err} at {self._index = }")
            return

        if timestamp - session_time > 0:
            sleep(timestamp - session_time)

        #print(timestamp, session_time)

        self._index += 1