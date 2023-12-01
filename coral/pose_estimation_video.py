# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage: py [-3.7 | -3.8 | -3.9] pose_estimation_video.py

Modified by: Mike Smith
"""

"""
movenet landmarks
0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
9: 'left_wrist', 10 : 'right_wrist', 11: 'left_hip', 12: 'right_hip', 
13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
"""

import time

from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

import cv2 as cv
from statistics import mean


NUM_KEYPOINTS = 17

VIS_THRESH = 0.5

KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def display_frame_rate(img, frame_times):
    frame_times["curr time"] = time.time()
    frame_rate = 1 / (frame_times["curr time"] - frame_times["prev time"])
    frame_times["prev time"] = frame_times["curr time"]

    img = cv.flip(img, 1)
    cv.putText(img, str(int(frame_rate)), (10, 30), 0, 1, (255, 255, 0))
    img = cv.flip(img, 1)
    return img


def main():
    model_path = "models/movenet/movenet_single_pose_thunder_ptq_edgetpu.tflite"
    #model_path = "models/posenet/mobilenet/posenet_mobilenet_v1_075_481_641_16_quant_decoder_edgetpu.tflite"
    #model_path = "models/posenet/resnet/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite"

    """ create interpreter and load model """
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    """ open video or webcam """
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    #cap = cv.VideoCapture("path_to_video.mp4")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    """ frame rate (for debugging) """
    frame_times = {"curr time": 0, "prev time": 0}

    x_mean, y_mean = -1, -1
    lpf_buf_x = {key: list() for key in range(NUM_KEYPOINTS)}
    lpf_buf_y = {key: list() for key in range(NUM_KEYPOINTS)}

    lpf_buf_len = 5
    lpf_mean_len = 5*lpf_buf_len
    lpf_buf_x_mean = {key: list() for key in range(NUM_KEYPOINTS)}
    lpf_buf_y_mean = {key: list() for key in range(NUM_KEYPOINTS)}
    
    """ main while loop """
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            print("can't connect to camera :(")
            break

        """ display frame rate """
        img = display_frame_rate(img, frame_times)

        """ resize image and perform antialiasing using cv.INTER_AREA """
        interpreter_size = common.input_size(interpreter)
        resized_img = cv.resize(img, (interpreter_size[0], interpreter_size[1]), interpolation=cv.INTER_AREA)
        common.set_input(interpreter, resized_img)
        interpreter.invoke()

        """ get inference results """
        pose = common.output_tensor(interpreter, 0).copy()
        pose = pose.reshape(NUM_KEYPOINTS, 3)
        lpf_pose = list()
        
        """ perform processing on raw data """
        h, w, _ = img.shape
        for i, lm in enumerate(pose):
            y, x, vis = lm

            if vis > VIS_THRESH:
                lpf_buf_x[i].append(x)
                lpf_buf_y[i].append(y)

            if len(lpf_buf_x[i]) >= lpf_buf_len:
                lpf_buf_x[i] = lpf_buf_x[i][-lpf_buf_len:]
                x_mean = mean(lpf_buf_x[i])
                
                lpf_buf_x_mean[i].append(x_mean)
                if len(lpf_buf_x_mean[i]) >= lpf_mean_len:
                    lpf_buf_x_mean[i] = lpf_buf_x_mean[i][-lpf_mean_len:]

            if len(lpf_buf_y[i]) >= lpf_buf_len:
                lpf_buf_y[i] = lpf_buf_y[i][-lpf_buf_len:]
                y_mean = mean(lpf_buf_y[i])

                lpf_buf_y_mean[i].append(y_mean)
                if len(lpf_buf_y_mean[i]) >= lpf_mean_len:
                    lpf_buf_y_mean[i] = lpf_buf_y_mean[i][-lpf_mean_len:]
                
            """ show landmark points """
            if len(set(lpf_buf_x_mean[i])) != 1 and len(set(lpf_buf_x_mean[i])) != 1:
                cv.circle(img, (int(x_mean*w), int(y_mean*h)), 8, (0, 0, 255), cv.FILLED)
                lpf_pose.append([int(x_mean*w), int(y_mean*h)])
            else:
                lpf_pose.append([-1, -1])

        if len(lpf_pose) != 0:

            """ show landmark connections """
            for i, (start, end) in enumerate(KEYPOINT_CONNECTIONS):

                start_x, start_y = int(lpf_pose[start][0]), int(lpf_pose[start][1])
                end_x, end_y = int(lpf_pose[end][0]), int(lpf_pose[end][1])

                if -1 not in [start_x, start_y, end_x, end_y]:
                    cv.line(img, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

        """ flip frame horizontally """
        img = cv.flip(img, 1)

        """ show image """
        cv.imshow("Image", img)

        """ press "q" to exit program """
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
  main()
