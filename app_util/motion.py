"""
motion.py

Motion tracking module. 
Handles the passing of information between the "MediaPipe Pose Estimation" library 
and the main application.

 -  Specifically for this project, motion tracking will require a 90% high confidence level 
    for detection.
 -  Continuous tracking will require a 50% confidence level. 
 -  This is to ensure the program is sure of the subject prior to tracking, 
    but stay locked on to the subject while tracking.

see "doc/motion.md" for more details

"""

import sys
import math
import numpy as np
import cv2 as cv
import mediapipe as mp
from statistics import mean
from .util import Util


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "12/04/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen", 
    "Live Myklebust", 
    "Amber Spurway",
]   


class Motion():
    """
    motion capture module

    """

    """ motion capture parameters """
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.3

    """ tracking id's """
    left_shoulder = 11
    right_shoulder = 12
    left_elbow = 13
    right_elbow = 14
    left_wrist = 15
    right_wrist = 16
    left_hip = 23
    right_hip = 24
    left_knee = 25
    right_knee = 26
    left_ankle = 27
    right_ankle = 28

    left_heel = 29
    right_heel = 30
    left_toes = 31
    right_toes = 32

    """ landmark key values """
    id = 0
    x = 1
    y = 2
    z = 3
    vis = 4

    crop_begin = Util.INIT
    crop_end = Util.INIT

    cropped = False

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ):
        self._pose_param_dict = {
            "mp pose": mp.solutions.pose,
            "mp draw": mp.solutions.drawing_utils,
        }
        self._pose = self._pose_param_dict["mp pose"].Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._prev_time = None
        self._prev_landmarks = None

        self._vel_max_buf_size = 5
        self._vel_buf = list()

        self._prev_crop = list()

        """ preset crop sequence for bbox """
        self._bbox_borders = {
            5: (0.08, 0.12),
            4: (0.16, 0.24),
            3: (0.24, 0.36),
            2: (0.32, 0.48),
            1: (0.40, 0.60),
        }

        """ number of crop options in sequence """
        self._crop_seq_len = max(self._bbox_borders.keys())

        """ for use when dynamic mode is disabled """
        self._border = self._bbox_borders[max(self._bbox_borders.keys())]

        """ buffer for lpf """
        self._lpf_buf_x = {key: list() for key in range(33)}
        self._lpf_buf_y = {key: list() for key in range(33)}
        self._lpf_buf_z = {key: list() for key in range(33)}
        self._lpf_buf_len = 3

    def track_motion(self, img, landmarks, curr_time, blur_faces, debug=False, dynamic=True, filter=True):

        """ disable dynamic bbox (crop) is detection is not consistent """
        if not all(self._prev_crop) and len(self._prev_crop) == self._crop_seq_len:
            dynamic = False

        """ reset crop list when no subject is detected """
        if not any(self._prev_crop):
            self._prev_crop = list()

        """ disable frame crop is dynamic bbox is disabled """
        if not dynamic:
            self.cropped = False

        """ append curr crop status to the prev crop list """
        self._prev_crop.append(self.cropped)
        if len(self._prev_crop) > self._crop_seq_len:
            self._prev_crop = self._prev_crop[-self._crop_seq_len:]

        """ change border dimensions only if dynamic mode is enabled """
        if dynamic:
            self._border = self._bbox_borders.get(len(self._prev_crop))

        """
        used for tracking motion within a bounding box

        """
        self._landmark_x_min = -1
        self._landmark_x_max = -1

        lm_pixels = []

        """ 
        crop frame based on the position of the detected person 
        in the previous frame
        
        """
        height, width, _ = img.shape
        if self.cropped:
            img_crop = img[
                self.crop_begin[Util.Y] : self.crop_end[Util.Y],
                self.crop_begin[Util.X] : self.crop_end[Util.X],
            ]

            """ calculate positional adjustment needed for the cropped frame """
            adjust_x, adjust_y, grad_x, grad_y = get_adjustments(
                self.crop_begin, self.crop_end, (height, width),
            )
        else:
            img_crop = img

        """ look for human motion in the bounding box / cropped frame """
        try:
            img_crop = cv.cvtColor(img_crop, cv.COLOR_BGR2RGB)
            self._results = self._pose.process(img_crop)
            img_crop = cv.cvtColor(img_crop, cv.COLOR_RGB2BGR)
        except:
            self.cropped = False
            return img, self.crop_begin, self.crop_end, self.cropped, None

        """ if human motion is detected """
        if self._results.pose_landmarks:

            for id, landmark in enumerate(self._results.pose_landmarks.landmark):
                
                """ apply positional adjustment for the cropped frame """
                if self.cropped:
                    landmark.x, landmark.y = apply_adjustments(
                        (landmark.x, landmark.y), (adjust_x, adjust_y), (grad_x, grad_y),
                    )

                x, y, z = int(landmark.x * width), int(landmark.y * height), int(landmark.z * 100)
                vis = int(landmark.visibility * 100)

                """ apply lpf """
                if filter:
                    self._lpf_buf_x[id].append(x)
                    self._lpf_buf_y[id].append(y)
                    self._lpf_buf_z[id].append(z)

                    if len(self._lpf_buf_x[id]) > self._lpf_buf_len:
                        self._lpf_buf_x[id] = self._lpf_buf_x[id][-self._lpf_buf_len:]
                        x = int(mean(self._lpf_buf_x[id]))

                    if len(self._lpf_buf_y[id]) > self._lpf_buf_len:
                        self._lpf_buf_y[id] = self._lpf_buf_y[id][-self._lpf_buf_len:]
                        y = int(mean(self._lpf_buf_y[id]))

                    if len(self._lpf_buf_z[id]) > self._lpf_buf_len:
                        self._lpf_buf_z[id] = self._lpf_buf_z[id][-self._lpf_buf_len:]
                        z = int(mean(self._lpf_buf_z[id]))

                """ 
                get the velocity of each landmark 
                
                """
                try:
                    if self._prev_landmarks is not None:
                        x_prev, y_prev, z_prev = self._prev_landmarks[id][1:4]
                        dx, dy, dz, dt = x - x_prev, y - y_prev, z - z_prev, curr_time - self._prev_time
                        vel_x, vel_y, vel_z = dx/dt, dy/dt, dz/dt

                        self._vel_buf.append(math.sqrt(vel_x**2 + vel_y**2 + vel_z**2))

                        if len(self._vel_buf) > self._vel_max_buf_size:
                            self._vel_buf = self._vel_buf[-self._vel_max_buf_size:]

                        vel = round(mean(self._vel_buf))

                    else:
                        vel = 0
                except:
                    vel = 0

                """ append raw co-ordinate values (pixel values) """
                lm = (id, x, y, z, vis, vel)
                landmarks.append(lm)

                """ 
                calculate co-ordinate values in pixels to be used later for 
                drawing the bounding box and to crop the next frame
                
                """
                lm_pixels.append((int(landmark.x * width), int(landmark.y * height)))

            """ face blurring """
            if blur_faces:
                x_lst, y_lst = list(), list()
                for id, x, y, z, vis, vel in landmarks:
                    if id in [i for i in range(11)]:
                        x_lst.append(x)
                        y_lst.append(y)

                rect_min_x = int(min(x_lst) - 0.05*width)
                rect_min_y = int(min(y_lst) - 0.1*height)
                rect_max_x = int(max(x_lst) + 0.05*width)
                rect_max_y = int(max(y_lst) + 0.1*height)

                img = circular_blur(img, rect_min_x, rect_max_x, rect_min_y, rect_max_y)

            """ draw the bounding box """
            self._landmark_x_min = min([lm[Util.X] for lm in lm_pixels]) - int(width*self._border[Util.X])
            self._landmark_x_max = max([lm[Util.X] for lm in lm_pixels]) + int(width*self._border[Util.X])
            self._landmark_y_min = min([lm[Util.Y] for lm in lm_pixels]) - int(height*self._border[Util.Y])
            self._landmark_y_max = max([lm[Util.Y] for lm in lm_pixels]) + int(height*self._border[Util.Y])

            x_min = max([self._landmark_x_min, 0])
            y_min = max([self._landmark_y_min, 0])
            x_max = min([self._landmark_x_max, width])
            y_max = min([self._landmark_y_max, height])

            start = (x_min + Util.BORDER_WIDTH, y_min + Util.BORDER_WIDTH)
            end = (x_max - Util.BORDER_WIDTH, y_max - Util.BORDER_WIDTH)

            self.crop_begin = (x_min, y_min)
            self.crop_end = (x_max, y_max)

            self.cropped = True

        else:
            self.cropped = False

        view = self.get_camera_view(img, landmarks)

        self._prev_time = curr_time
        self._prev_landmarks = landmarks.copy() if len(landmarks) > 0 else None

        """ returns image frame """
        return img, self.crop_begin, self.crop_end, self.cropped, view

    def get_camera_view(self, img, landmarks):

        h, w, _ = img.shape

        """ camera angle """
        if len(landmarks) != 0:
            if abs(landmarks[self.left_hip][3] - landmarks[self.right_hip][3]) < 15:
                if landmarks[self.left_hip][1] < landmarks[self.right_hip][1]:
                    view = Util.REAR_VIEW
                else:
                    view = Util.FRONT_VIEW
            else:
                view = Util.SIDE_VIEW
        else:
            view = None

        return view
    
    def get_lm_vel(self, landmarks):

        if self._prev_landmarks is None:
            return None
        
        for lm, prev_lm in zip(landmarks, self._prev_landmarks):
            pass
    
    def draw(self, img, landmarks, crop_begin, crop_end):
        """
        overlays the detected landmarks as a stick figure onto the frame
        
        """
        if self._results.pose_landmarks:

            self.draw_connections(
                img, landmarks, self._pose_param_dict["mp pose"].POSE_CONNECTIONS
            )

            """ highlight important points """
            for id, x, y, z, vis, vel in landmarks:

                left = [
                    self.left_wrist, self.left_elbow, self.left_shoulder,
                    self.left_hip, self.left_knee, self.left_ankle,
                ]
                right = [
                    self.right_wrist, self.right_elbow, self.right_shoulder,
                    self.right_hip, self.right_knee, self.right_ankle,
                ]
                if id in left:
                    cv.circle(img, (x, y), int(8), Util.GREEN, cv.FILLED)
                if id in right:
                    cv.circle(img, (x, y), int(8), Util.RED, cv.FILLED)

        cv.rectangle(img, crop_begin, crop_end, Util.BLUE, Util.BORDER_WIDTH)

    def draw_connections(self, img, landmarks, connections):
        """
        draws the connections between points

        """
        for start, end in connections:
            try:
                start_x, start_y = landmarks[start][1:3]
                end_x, end_y = landmarks[end][1:3]
                cv.line(img, (start_x, start_y), (end_x, end_y), Util.WHITE, 2)
            except IndexError as err:
                print(err)
        

class Hand():
    """
    hand tracking module
    
    """

    """ hand tracking parameters """
    max_num_hands = 1
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    wrist = 0
    thumb = 4
    index = 8
    middle = 12
    ring = 16
    pinky = 20

    def __init__(self, 
        static_image_mode = False,
        max_num_hands = max_num_hands,
        model_complexity = 1,
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence,
    ):
        self._hands_param_dict = {
            "mp hands": mp.solutions.hands,
            "mp draw": mp.solutions.drawing_utils
        }
        self._hands = self._hands_param_dict["mp hands"].Hands(
            static_image_mode = static_image_mode,
            max_num_hands = max_num_hands,
            model_complexity = model_complexity,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

        self._max_array_len = 10
        self._left_or_right_array = list()
        self._left_or_right = None

    def track_hands(self, img, landmarks, begin, end, source):
        """
        detects hands in the frame

        """
        height, width, _ = img.shape
        img_crop = img[
            begin[Util.Y]:end[Util.Y], begin[Util.X]:end[Util.X],
        ]

        """ calculate positional adjustment needed for the cropped frame """
        adjust_x, adjust_y, grad_x, grad_y = get_adjustments(
            begin, end, (height, width),
        )

        img_crop = cv.cvtColor(img_crop, cv.COLOR_BGR2RGB)
        results = self._hands.process(img_crop)
        img_crop = cv.cvtColor(img_crop, cv.COLOR_RGB2BGR)

        height, width, _ = img.shape
        if results.multi_hand_landmarks:

            for i, handedness in enumerate(results.multi_handedness):

                """ check for left or right handedness """
                if handedness.classification[0].score > 0.98:
                    self._left_or_right_array.append(handedness.classification[0].label)

                if len(self._left_or_right_array) > self._max_array_len:
                    self._left_or_right_array = self._left_or_right_array[-self._max_array_len:]

                    # for the flipped frame
                    right_count = self._left_or_right_array.count(Util.LEFT)
                    left_count = self._left_or_right_array.count(Util.RIGHT)

                    if left_count == self._max_array_len:
                        self._left_or_right = Util.LEFT
                    elif right_count == self._max_array_len:
                        self._left_or_right = Util.RIGHT
                    else:
                        self._left_or_right = None

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                """ extract all landmarks for each detected hand """
                for id, landmark in enumerate(hand_landmarks.landmark):

                    landmark.x, landmark.y = apply_adjustments(
                        (landmark.x, landmark.y), (adjust_x, adjust_y), (grad_x, grad_y),
                    )

                    lm = (
                        id, int(landmark.x * width), int(landmark.y * height),
                        int(landmark.z * 200), int(landmark.visibility * 100),
                    )
                    landmarks.append(lm)

                    finger_params = [
                        self.thumb, self.index, self.middle, self.ring, self.pinky,
                    ]
                    if id in finger_params:
                        cv.circle(img, (landmarks[id][1], landmarks[id][2]), 10, Util.CYAN, 2)

                    if self._left_or_right is not None and source == Util.VIDEO:
                        pos = (landmarks[self.wrist][1], landmarks[self.wrist][2])
                        font = cv.FONT_HERSHEY_SIMPLEX
                        cv.putText(img, self._left_or_right, pos, font, 0.8, Util.BLUE, 2)

        return img, self._left_or_right
    

class Faces():

    def __init__(self,
        static_image_mode = False,
        max_num_faces = 128,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.3,
    ):
        self._faces_param_dict = {
            "mp faces": mp.solutions.face_mesh,
            "mp draw": mp.solutions.drawing_utils
        }
        self._faces = self._faces_param_dict["mp faces"].FaceMesh(
            static_image_mode = static_image_mode,
            max_num_faces = max_num_faces,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
        )
        self._draw_spec = self._faces_param_dict["mp draw"].DrawingSpec(
            thickness = 1, circle_radius = 1
        )
        self._start = (0, 0)
        self._end = (0, 0)

    def find_faces(self, img): #, landmarks, crop):
        h, w, _ = img.shape

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self._faces.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        self._start, self._end = (0, 0), (0, 0)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x, y = list(), list()
                for id, landmark in enumerate(face_landmarks.landmark):

                    if id == 10 or id == 152 or id == 162 or id == 389:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        x.append(cx)
                        y.append(cy)

                rect_min_x, rect_min_y = (int(min(x) - 0.05*w), int(min(y) - 0.1*h))
                rect_max_x, rect_max_y = (int(max(x) + 0.05*w), int(max(y) + 0.1*h))

                img = circular_blur(img, rect_min_x, rect_max_x, rect_min_y, rect_max_y)

        return img


def get_adjustments(begin, end, dim):
    """
    returns the positional adjustments based on the current frame crop

    """
    h, w = dim

    adjust_x = begin[Util.X] / w
    adjust_y = begin[Util.Y] / h

    grad_x = (end[Util.X] - begin[Util.X]) / w
    grad_y = (end[Util.Y] - begin[Util.Y]) / h

    return adjust_x, adjust_y, grad_x, grad_y


def apply_adjustments(lm, adjust, grad):
    """
    takes the adjustment values and apply to all landmark co-ordinates for current frame

    """
    x, y = lm

    x = grad[Util.X] * x + adjust[Util.X]
    y = grad[Util.Y] * y + adjust[Util.Y]

    return x, y


def circular_blur(img, *args):
    """
    draws a circular blurred ares in the frame
    used to blur out faces for privacy reasons

    """
    min_x, max_x, min_y, max_y = args

    try:
        roi = img[min_y:max_y, min_x:max_x]     # gets the region of interest based on face position
        blur = cv.medianBlur(roi, 99)           # applies blur locally to rectangular roi

        mask = np.zeros_like(roi)               # create a mask: same shape as the roi
        mask_h, mask_w = mask.shape[0:2]        # get the h and w of the mask
        centre = mask_w // 2, mask_h // 2       # get the centre point of the mask
        radii = int(min([mask_h, mask_w]) // 2) # get the closest distance from centre to edge of mask

        mask = cv.circle(mask, centre, radii, Util.WHITE, -1)   # draw a white circle on the mask
        mask = cv.bitwise_and(blur, mask)       # 'bitwise and' the circular mask and the rectangular roi

        roi = cv.circle(roi, centre, radii, Util.BLACK, -1)     # draw a black circle on the og roi 
        roi = cv.bitwise_or(roi, mask)          # 'bitwise or' the blurred mask and the blacked-out roi
        img[min_y:max_y, min_x:max_x] = roi     # replace rectangular roi with new roi that contains blur
    
    except Exception as err:
        pass
        
    return img