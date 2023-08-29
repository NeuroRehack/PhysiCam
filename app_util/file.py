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
    supported_files = {Util.CSV: ".csv", Util.MP4: ".mp4", Util.AVI: ".avi"}

    def __init__(self, save=True):
        """
        save: a boolean to specify whether or not to generate a file

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
    

class CsvFile(File):
    """
    csv file module: contains specific method for interfacing with csv files

    """

    def __init__(self, save=True):

        super().__init__(save)

        self._data = []
        self._prev_time = 0
    
    def set_save_status(self, save):
        """
        set whether or not to generate a file

        """
        self._save_file = save

    def read(self):
        """
        not implemented yet

        """
        pass

    def write(self, name, shape):
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
        # TO-DO: move invalid char checking to line-edit callback function
        invalid = r'/\:*"?<>|!@#$%^&'

        """ create an appropriate filename (display to terminal for debugging) """
        name = "" if any(c in name for c in invalid) else name
        name = f"{name}-" if name != "" else ""
        name = name.replace(' ', '_')

        fname = f"{self.file_path}/{name}{self.create_filename()}" 
        print(f"saved file: {fname}") if fname not in files else print("file exists")

        """ create a csv file and write to it """
        with open(fname, "w", newline="") as new_file:
            dict_writer = csv.DictWriter(new_file, fieldnames=self._keys)
            dict_writer.writeheader()

            for d in self._data:
                dict_writer.writerow(d)

    def create_filename(self):
        """
        creates a unique filename using the current system time and date

        """
        return f'{time.strftime("%y%m%d-%H%M%S")}.csv'

    def parse_movements(self, movements, landmarks, curr_time, shape, corr_mode=False):
        """
        parses the movement data periodically and stores it to be written laters

        """
        if not self._save_file:
            return
        
        delay = 0.01 if corr_mode else 0.05

        """ init field names for csv file """
        if self._prev_time == 0:

            self._prev_time = 0 if corr_mode else time.time()

            self._keys = [
                key for key in movements.keys() if movements[key].get_tracking_status()
            ]
            self._keys.insert(0, "system time")
            self._keys.insert(1, "time")
            self._keys.insert(2, "resolution")
            self._keys.insert(3, "")
            self._keys.append("")

            for i in range(33):
                self._keys.append(i)

        """ update data every ~100ms """
        if time.time() > self._prev_time + delay:
            data = {
                "system time": f'{datetime.now().strftime("%y/%m/%d_%H:%M:%S.%f")[:-3]}',
                "time": "%d:%02d:%02d.%02d"
                % (
                    curr_time // 3600,
                    curr_time // 60,
                    curr_time % 60,
                    (curr_time % 1) * 100,
                ),
                "resolution": shape,
            }
            for key, value in movements.items():
                if movements[key].get_tracking_status():
                    data[key] = value.get_count()

            if len(landmarks) > 0:
                for i, (id, x, y, z, vis, vel) in enumerate(landmarks):
                    data[i] = (x, y, z, vis, vel)

            self._data.append(data)
            self._prev_time = time.time()