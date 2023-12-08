"""
file.py

Handles reading and writing files. 
Contains methods for checking invalid files, parsing movement data and writing to csv files.

Supported files:
- `.csv`
- `.mp4`

see "doc/file.md" for more details

"""

import csv
import os
import time
import cv2 as cv
from .util import Util
from datetime import datetime


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "01/05/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen", 
    "Live Myklebust", 
    "Amber Spurway",
]


class File:
    """
    file module: handles reading and writing files

    """

    """ default file path (in sub-dir "files" located in current dir) """
    file_path = Util.DEFAULT_FILE_PATH
    video_path = Util.DEFAULT_VIDEO_PATH
    supported_files = {Util.CSV: ".csv", Util.MP4: ".mp4", Util.AVI: ".avi"}

    def __init__(self, save=True):
        """
        save: a boolean to specify whether or not to generate a file

        """
        self._save_file = save

    def set_save_status(self, save):
        """
        set whether or not to generate a file

        """
        self._save_file = save

    def get_file_type(self, filename):
        """
        checks that the file is supported by the program
        returns the file type

        """
        for file_type, file_ext in self.supported_files.items():
            if file_ext in filename:
                return file_type

        if filename == "":
            return

        return Util.FILE_NOT_SUPPORTED
    

class VideoFile(File):
    """
    class for parsing and saving video frames
    only to be used in testing

    """
    def __init__(self, save=True):
        """
        init method

        """
        super().__init__(save)

        self._video_out = None
        self._timestamps = list()

    def start_video(self, filetime, shape, cam_id):
        """
        start saving to a new video file

        """
        h, w, _ = shape

        if not self._save_file:
            return

        """ create a directory to store the saved files """
        if not os.path.exists(self.video_path):
            os.mkdir(self.video_path)

        """ create a sub-dir to store the video file and timestamps file """
        if not os.path.exists(f"{self.video_path}/{filetime}-{cam_id}"):
            os.mkdir(f"{self.video_path}/{filetime}-{cam_id}")

        """ create the video writer object """
        self._fname = f"{self.video_path}/{filetime}-{cam_id}/{filetime}-{cam_id}"
        self._video_out = cv.VideoWriter(
            f"{self._fname}.avi", cv.VideoWriter_fourcc(*'XVID'), 30, (w, h),
        )

        """ init the timestamps list """
        self._timestamps = list()

    def parse_video_frame(self, frame, curr_time):
        """
        method to parse and save each frame of the video
        also saves the corresponding timestamp to the timestamps file
        
        """
        if not self._save_file:
            return

        if curr_time is None:
            return
        try:
            self._video_out.write(frame)
            self._timestamps.append(curr_time)
        except:
            pass

    def end_video(self):
        """
        method to handle the end of saving a video
        
        """
        if not self._save_file:
            return

        self._video_out.release()

        # write the timestamps to a text file at the end of the recording
        with open(f"{self._fname}.txt", "a") as f:
            for t in self._timestamps:
                f.write(f"{t}\n")

        print(f"saved video: {self._fname}.avi")
        print(f"saved timestamps file: {self._fname}.txt")
                

class CsvFile(File):
    """
    csv file module: contains specific method for interfacing with csv files

    """

    def __init__(self, save=True):

        super().__init__(save)

        self._data = []
        self._prev_time = 0

    def read(self):
        """
        not implemented yet

        """
        pass

    def write(self, name, filetime, cam_id):
        """
        takes the parsed data and writes it to a csv file

        """
        if not self._save_file or len(self._data) == 0:
            return

        """ create a directory to store the saved files """
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

        """ make sure there are no existing files with duplicate names """
        dir = os.scandir(self.file_path)
        files = [a.name for a in dir]

        """ invalid characters """
        invalid = r'/\:*"?<>|!@#$%^&'

        """ create an appropriate filename (display to terminal for debugging) """
        name = "" if any(c in name for c in invalid) else name
        name = f"{name}-" if name != "" else ""
        name = name.replace(' ', '_')

        fname = f"{self.file_path}/{name}{filetime}-{cam_id}.csv"
        print(f"saved file: {fname}") if fname not in files else print("file exists")

        """ create a csv file and write to it """
        with open(fname, "w", newline="") as new_file:
            dict_writer = csv.DictWriter(new_file, fieldnames=self._keys)
            dict_writer.writeheader()

            for d in self._data:
                dict_writer.writerow(d)

    def parse_movements(
            self, movements, landmarks, curr_time, shape, curr_movement, flipped,
            corr_mode=False,
        ):
        """
        parses the movement data periodically and stores it to be written laters

        """
        if not self._save_file:
            return
        
        delay = 0 if corr_mode else 0

        """ init field names for csv file """
        if self._prev_time == 0:

            self._prev_time = 0 if corr_mode else time.time()

            self._keys = [
                key for key in movements.keys() # if movements[key].get_tracking_status()
            ]
            self._keys.insert(0, "system time")
            self._keys.insert(1, "time")
            self._keys.insert(2, "resolution")
            self._keys.insert(3, "")
            self._keys.insert(4, "current movement")
            self._keys.insert(5, "frame flipped")
            self._keys.insert(6, "")
            self._keys.append("")

            for i in range(33):
                self._keys.append(i)

        """ update data """
        if time.time() > self._prev_time + delay:
            data = {
                "system time": f'{datetime.now().strftime("%y/%m/%d_%H:%M:%S.%f")[:-3]}',
                "time": "%d:%02d:%02d.%05d"
                % (
                    curr_time // 3600,
                    curr_time // 60,
                    curr_time % 60,
                    (curr_time % 1) * 1e5,
                ),
                "resolution": shape,
                "current movement": curr_movement,
                "frame flipped": 1 if flipped else 0,
            }
            for key, value in movements.items():
                data[key] = None

                if movements[key].get_tracking_status():
                    data[key] = value.get_count()

            if len(landmarks) > 0:
                for i, (id, x, y, z, vis, vel) in enumerate(landmarks):
                    data[i] = (x, y, z, vis, vel)

            self._data.append(data)
            self._prev_time = time.time()