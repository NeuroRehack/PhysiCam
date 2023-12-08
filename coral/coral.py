"""
coral/coral.py

library for interfacing with the coral tpu usb accelerator

the Coral module contain the following methods:
- `__init__`: sets up the tpu interpreter, loads ml model, init lpf arrays
- `get_landmarks`: takes a video frame as input and outputs pose landmark co-ords
- `display_landmarks`: draws the detected landmarks on the video frame

"""

__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "05/12/2023"
__status__ = "Prototype"
__credits__ = []


import cv2 as cv
import numpy as np
from statistics import mean
from traceback import format_exc
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


class Coral():

    num_keypoints = 17

    keypoint_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    model_path = "coral/models/movenet/movenet_single_pose_thunder_ptq_edgetpu.tflite"
    #model_path = "coral/models/posenet/mobilenet/posenet_mobilenet_v1_075_481_641_16_quant_decoder_edgetpu.tflite"

    def __init__(self):
        """
        - sets up the tpu interpreter
        - loads ml model
        - init lpf arrays

        """
        try:
            self._interpreter = make_interpreter(self.model_path)
            self._interpreter.allocate_tensors()
        except Exception as err:
            print(f"{format_exc()}")
            raise ValueError

        self._x_mean, self._y_mean = -1, -1
        self._lpf_buf_x = {key: list() for key in range(self.num_keypoints)}
        self._lpf_buf_y = {key: list() for key in range(self.num_keypoints)}

        self._lpf_buf_len = 5
        self._lpf_mean_len = 5*self._lpf_buf_len
        self._lpf_buf_x_mean = {key: list() for key in range(self.num_keypoints)}
        self._lpf_buf_y_mean = {key: list() for key in range(self.num_keypoints)}

        self._print_once = True
    
    def get_landmarks(self, img):
        """
        takes a video frame as input and outputs pose landmark co-ords

        params:
        - `img`: the current video frame

        returns:
        - a list containing the pose landmark co-ords

        """
        interpreter_size = common.input_size(self._interpreter)

        """ resize image and perform antialiasing using cv.INTER_AREA """
        num = 5

        x = np.linspace(img.shape[0], interpreter_size[0], num)
        y = np.linspace(img.shape[1], interpreter_size[1], num)
        if self._print_once: print(x, y)

        resized_img = img
        for i in range(num):
            resized_img = cv.resize(
                img, (int(x[i]), int(y[i])), interpolation=cv.INTER_AREA
            )
            if self._print_once: print(resized_img.shape)

        #cv.imshow("resized", resized_img)
        #key = cv.waitKey(1)
        self._print_once = False

        common.set_input(self._interpreter, resized_img)
        self._interpreter.invoke()

        """ get inference results """
        self._landmarks = common.output_tensor(self._interpreter, 0).copy()
        self._landmarks = self._landmarks.reshape(self.num_keypoints, 3)
    
        self._lpf_pose = list()
        
        """ perform processing on raw data """
        h, w, _ = img.shape
        for i, lm in enumerate(self._landmarks):
            y, x, vis = lm

            if vis > 0.5:
                self._lpf_buf_x[i].append(x)
                self._lpf_buf_y[i].append(y)

                #print(self._lpf_buf_x[0])

            if len(self._lpf_buf_x[i]) >= self._lpf_buf_len:
                self._lpf_buf_x[i] = self._lpf_buf_x[i][-self._lpf_buf_len:]
                self._x_mean = mean(self._lpf_buf_x[i])
                
                self._lpf_buf_x_mean[i].append(self._x_mean)
                if len(self._lpf_buf_x_mean[i]) >= self._lpf_mean_len:
                    self._lpf_buf_x_mean[i] = self._lpf_buf_x_mean[i][-self._lpf_mean_len:]

            if len(self._lpf_buf_y[i]) >= self._lpf_buf_len:
                self._lpf_buf_y[i] = self._lpf_buf_y[i][-self._lpf_buf_len:]
                self._y_mean = mean(self._lpf_buf_y[i])

                self._lpf_buf_y_mean[i].append(self._y_mean)
                if len(self._lpf_buf_y_mean[i]) >= self._lpf_mean_len:
                    self._lpf_buf_y_mean[i] = self._lpf_buf_y_mean[i][-self._lpf_mean_len:]

        return [self._lpf_buf_x_mean, self._lpf_buf_y_mean]

    def display_landmarks(self, img, landmarks):
        """
        draws the detected landmarks on the video frame

        params:
        - `img`: the current video frame
        - `landmarks`: the detected pose landmark co-ords

        returns:
        - the current video frame with visual overlaid landmarks

        """
        h, w, _ = img.shape
        x, y = landmarks
        for i in range(self.num_keypoints):

            """ show landmark points """
            if len(set(x[i])) > 1 and len(set(y[i])) > 1:
                cv.circle(img, (int(x[i][-1]*w), int(y[i][-1]*h)), 8, (0, 0, 255), cv.FILLED)
                self._lpf_pose.append([int(x[i][-1]*w), int(y[i][-1]*h)])
            else:
                self._lpf_pose.append([-1, -1])

        if len(self._lpf_pose) != 0:

            """ show landmark connections """
            for i, (start, end) in enumerate(Coral.keypoint_connections):

                start_x, start_y = int(self._lpf_pose[start][0]), int(self._lpf_pose[start][1])
                end_x, end_y = int(self._lpf_pose[end][0]), int(self._lpf_pose[end][1])

                if -1 not in [start_x, start_y, end_x, end_y]:
                    cv.line(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

        return img

        