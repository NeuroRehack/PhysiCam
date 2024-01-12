"""
movement.py

Contains a generic movement class. 
Each movement is defined by a set of angle (each defined by three points) 
and a set of positional thresholds (each define by two points).

see "doc/movement.md" for more details

"""

import sys
import math
import time
import cv2 as cv
import numpy as np
from .motion import Motion, Hand
from .util import Util
from statistics import mean


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "23/04/2023"
__status__ = "Prototype"
__credits__ = [
    "Agnethe Kaasen",
    "Live Myklebust",
    "Amber Spurway",
]


class Generic:
    """
    generic movement function containing more general functions

    """
    def __init__(self, is_tracking, ignore_vis, debug):
        """
        is_tracking: a boolean to specify whether tracking is enabled

        ignore_vis: a boolean to ignore visibility thresholds
            set to "True" if tracking full-body / compound movements
        debug: a boolean to allow program to overlay angle values on frame

        """
        self._is_tracking = is_tracking
        self._ignore_vis = ignore_vis
        self._debug = debug
        
        self._last_count_time = 0

    def set_tracking_status(self, is_tracking):
        """
        set the tracking status
        can be used to turn on/off tracking for certain movements
        """
        self._is_tracking = is_tracking

    def get_tracking_status(self):
        """
        get the current tracking status
        returns True is tracking is active, else returns False
        """
        return self._is_tracking
    
    def find_angle_2d(self, x, y):
        """
        find angle between two vectors in 2d space

        """
        x1, x2, x3 = x
        y1, y2, y3 = y

        return abs(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))

    def find_angle_3d(self, x, y, z):
        """
        find angle between two vectors in 3d space

        """

        def adj_z(z):
            return tuple(0.3*a for a in z)

        x1, x2, x3 = x
        y1, y2, y3 = y
        z1, z2, z3 = adj_z(z)

        a1 = [x1 - x2, y1 - y2, z1 - z2]
        a2 = [x3 - x2, y3 - y2, z3 - z2]

        def mag(x):
            return math.sqrt(sum(i**2 for i in x))
        
        return abs(math.degrees(math.acos(np.dot(a1, a2) / (mag(a1) * mag(a2)))))

    def find_angle(self, shape, points, ignore_vis=None):
        """
        generic function for calutating the angle between two lines
        defined by three points

        note: p2 must be the common point between the two lines

        """
        if points is None:
            return -1

        if ignore_vis is None:
            ignore_vis = self._ignore_vis

        p1, p2, p3 = points

        _, x1, y1, z1, v1, __ = p1
        _, x2, y2, z2, v2, __ = p2
        _, x3, y3, z3, v3, __ = p3

        """
        check that the visibility of the points is above the visibility threshold
        otherwise, ignore points as this can lead to program guessing position of points

        """
        if ignore_vis:
            c0 = [True]
        else:
            c0 = [v1 > Util.VIS, v2 > Util.VIS, v3 > Util.VIS]

        """
        check that points are not too close to the edge of the frame
        otherwise, ignore points as this can lead to inaccurate angle values

        """
        h, w, _ = shape
        x_min, y_min = Util.MIN * w, Util.MIN * h
        x_max, y_max = Util.MAX * w, Util.MAX * h

        c1 = [x1 > x_min, x1 < x_max, y1 > y_min, y1 < y_max]
        c2 = [x2 > x_min, x2 < x_max, y2 > y_min, y2 < y_max]
        c3 = [x3 > x_min, x3 < x_max, y3 > y_min, y3 < y_max]

        if all(c0) and all(c1) and all(c2) and all(c3):

            try:
                angle = self.find_angle_3d((x1, x2, x3), (y1, y2, y3), (z1, z2, z3))
            except ValueError as err:

                # possible error at: angle = math.acos(np.dot(a1, a2) / (mag(a1) * mag(a2)))
                # ValueError: math domain error

                angle = self.find_angle_2d((x1, x2, x3), (y1, y2, y3))
                print(err)

        else:
            angle = -1

        """ make sure all angle values are between 0 and 180 degrees """
        if angle < 180:
            return angle
        else:
            return 360 - angle


    def find_gradient(self, shape, points, ignore_vis=None):
        """
        generic function for calculating the absolute gradient in a line joining two points

        """
        if points is None:
            return -1

        if ignore_vis is None:
            ignore_vis = self._ignore_vis

        p1, p2 = points

        _, x1, y1, z1, v1, __ = p1
        _, x2, y2, z2, v2, __ = p2

        """
        check that the visibility of the points is above the visibility threshold
        otherwise, ignore points as this can lead to program guessing position of points

        """
        if ignore_vis:
            c0 = [True]
        else:
            c0 = [v1 > Util.VIS, v2 > Util.VIS]

        h, w, _ = shape
        x_min, y_min = Util.MIN * w, Util.MIN * h
        x_max, y_max = Util.MAX * w, Util.MAX * h

        c1 = [x1 > x_min, x1 < x_max, y1 > y_min, y1 < y_max]
        c2 = [x2 > x_min, x2 < x_max, y2 > y_min, y2 < y_max]

        if all(c0) and all(c1) and all(c2):
            try:
                grad = round(abs(y2 - y1) / abs(x2 - x1), 3)
            except ZeroDivisionError as err:
                grad = sys.maxsize
        else:
            grad = -1

        return grad

    def get_position(self, pos, landmarks, x_or_y):
        """
        returns the current x or y position of specifies landmark
        pos: the index position of the specified landmark (same ad the landmark id)
        landmarks: a list of all tracking landmarks
        x_or_y: whether to get the x or y co-ordinate of the given point

        """
        return landmarks[pos][x_or_y]
    
    def annotate(self, img, source, landmarks, angle, index=None, lm=None, type=None):
        """
        overlays angle values onto video frames

        """
        h, w, _ = img.shape

        if index is not None:
            angle = round(angle["curr"])

        colour = Util.RED if angle < 0 else Util.GREEN
        font = cv.FONT_HERSHEY_SIMPLEX

        if lm is not None:
            id, x, y, z, vis, vel = landmarks[lm]
        else:
            id, x, y, z, vis, vel = landmarks[self._points[index][1]]

        if index is not None:
            if source == Util.WEBCAM:
                img = cv.flip(img, 1)
                x = w - x

                cv.putText(img, str(angle), (x, y), font, 0.8, colour, 2)
                img = cv.flip(img, 1)

            else:
                cv.putText(img, str(angle), (x, y), font, 0.8, colour, 2)

        elif lm is not None:
            if type == "angle":
                angle = f"a: {round(angle)}"
            elif type == "grad" and angle >= 0:
                colour = Util.CYAN
                angle = f"g: {angle}"
            elif type == "vel":
                colour = Util.GREEN if angle > 100 else Util.RED
                angle = f"v: {angle}"

            if source == Util.WEBCAM:
                img = cv.flip(img, 1)
                x = w - x

                cv.putText(img, str(angle), (x, y), font, 0.8, colour, 2)
                img = cv.flip(img, 1)

            else:
                cv.putText(img, str(angle), (x, y), font, 0.8, colour, 2)

        return img
    
    def reset_count(self):
        """
        init count to zero by default
        """
        self._count = 0
        self._frame = list()
        self._reset = False

    def get_count(self):
        """
        returns the current movement count values

        """
        return self._count
    
    def get_last_count_time(self):
        return self._last_count_time
    
    def increment_count(self):
        self._count += 1
        self._last_count_time = time.time()
    
    def decrement_count(self, id):
        self._count -= 1
        print(id)
    

class ArmExtensions(Generic):
    """
    arm extensions

    """
    left_arm_reach_elbow_angle = 130
    right_arm_reach_elbow_angle = 130

    left_arm_reach_shoulder_angle = 50
    right_arm_reach_shoulder_angle = 50

    def __init__(self, is_tracking, left_or_right, ignore_vis=False, debug=False):
        """
        left_or_right: specify left or right side

        """
        super().__init__(is_tracking, ignore_vis, debug)
        self._left_or_right = left_or_right

        self._frame_len = 5
        self.reset_count()

        self._left_elbow_angle_points = (Motion.left_wrist, Motion.left_elbow, Motion.left_shoulder)
        self._left_shoulder_angle_points = (Motion.left_elbow, Motion.left_shoulder, Motion.left_hip)

        self._right_elbow_angle_points = (Motion.right_wrist, Motion.right_elbow, Motion.right_shoulder)
        self._right_shoulder_angle_points = (Motion.right_elbow, Motion.right_shoulder, Motion.right_hip)

        self._vel_frame = list()
        self._vel_frame_len = 5

    def track_movement(self, landmarks, img, source):
        """
        count the number of reps for the movement

        """
        self._elbow_thresh = {
            Util.RIGHT: self.right_arm_reach_elbow_angle, 
            Util.LEFT: self.left_arm_reach_elbow_angle,
        }
        self._shoulder_thresh = {
            Util.RIGHT: self.right_arm_reach_shoulder_angle,
            Util.LEFT: self.left_arm_reach_shoulder_angle,
        }

        if len(landmarks) > 0:

            if self._left_or_right == Util.LEFT:
                p_elbow = Util.get_points(landmarks, self._left_elbow_angle_points)
                p_shoulder = Util.get_points(landmarks, self._left_shoulder_angle_points)

            elif self._left_or_right == Util.RIGHT:
                p_elbow = Util.get_points(landmarks, self._right_elbow_angle_points)
                p_shoulder = Util.get_points(landmarks, self._right_shoulder_angle_points)
            
            else:
                p_elbow, p_shoulder = None, None

            elbow_angle = self.find_angle(img.shape, p_elbow, self._ignore_vis)
            shoulder_angle = self.find_angle(img.shape, p_shoulder, self._ignore_vis)

            if self._debug:
                img = self.debug(img, source, landmarks, elbow_angle, shoulder_angle)

            #if self._left_or_right == Util.LEFT:
                #print(self._elbow_thresh[self._left_or_right])

            if elbow_angle > 0 and shoulder_angle > 0:
                self._frame.append(
                    elbow_angle > self._elbow_thresh[self._left_or_right]
                    and shoulder_angle > self._shoulder_thresh[self._left_or_right]
                )

                if len(self._frame) > self._frame_len:
                    self._frame = self._frame[-self._frame_len:]

                    if all(self._frame) and self._reset:
                        self.increment_count()
                        self._reset = False

                    if not any(self._frame):
                        self._reset = True

        return img, self._count
    
    def debug(self, img, source, landmarks, *args):

        elbow_angle, shoulder_angle = args

        if self._left_or_right == Util.RIGHT:
            lm_elbow, lm_shoulder = Motion.right_elbow, Motion.right_shoulder
        elif self._left_or_right == Util.LEFT:
            lm_elbow, lm_shoulder = Motion.left_elbow, Motion.left_shoulder
        else:
            lm_elbow, lm_shoulder = None, None

        img = self.annotate(img, source, landmarks, elbow_angle, lm=lm_elbow, type="angle")
        img = self.annotate(img, source, landmarks, shoulder_angle, lm=lm_shoulder, type="angle")
        
        if self._left_or_right == Util.LEFT:
            wrist_vel = landmarks[Motion.left_wrist][5]
            shoulder_vel = landmarks[Motion.left_shoulder][5]
            lm = Motion.left_wrist
        elif self._left_or_right == Util.RIGHT:
            wrist_vel = landmarks[Motion.right_wrist][5]
            shoulder_vel = landmarks[Motion.right_shoulder][5]
            lm = Motion.right_wrist

        if wrist_vel is not None and shoulder_vel is not None:
            self._vel_frame.append(abs(wrist_vel - shoulder_vel))

        if len(self._vel_frame) > self._vel_frame_len:
            self._vel_frame = self._vel_frame[-self._vel_frame_len:]
            vel = round(mean(self._vel_frame))
            img = self.annotate(img, source, landmarks, vel, lm=lm, type="vel")

        return img
    
    def get_thresh(self):
        if self._left_or_right == Util.LEFT:
            return (self.left_arm_reach_elbow_angle, self.left_arm_reach_shoulder_angle)
        elif self._left_or_right == Util.RIGHT:
            return (self.right_arm_reach_elbow_angle, self.right_arm_reach_shoulder_angle)
    

class SitToStand(Generic):
    """
    sit-to-stand class

    """
    sit_to_stand_hip_angle = 130
    sit_to_stand_body_gradient = 5

    def __init__(self, is_tracking, ignore_vis=False, debug=False):

        super().__init__(is_tracking, ignore_vis, debug)

        self._frame_len = 10
        self.reset_count()

        self._right_hip_angle_points = (Motion.right_shoulder, Motion.right_hip, Motion.right_knee)
        self._left_hip_angle_points = (Motion.left_shoulder, Motion.left_hip, Motion.left_knee)

        self._right_body_grad_points = (Motion.right_shoulder, Motion.right_hip)
        self._left_body_grad_points = (Motion.left_shoulder, Motion.left_hip)

    def track_movement(self, landmarks, img, source):

        if len(landmarks) > 0:

            p = Util.get_points(landmarks, self._right_hip_angle_points)
            right_hip_angle = self.find_angle(img.shape, p, ignore_vis=True)

            p = Util.get_points(landmarks, self._left_hip_angle_points)
            left_hip_angle = self.find_angle(img.shape, p, ignore_vis=True)

            p = Util.get_points(landmarks, self._right_body_grad_points)
            right_body_grad = self.find_gradient(img.shape, p, ignore_vis=self._ignore_vis)

            p = Util.get_points(landmarks, self._left_body_grad_points)
            left_body_grad = self.find_gradient(img.shape, p, ignore_vis=self._ignore_vis,)

            if right_hip_angle > 0 and left_hip_angle > 0:
                self._frame.append(
                    right_hip_angle > self.sit_to_stand_hip_angle or left_hip_angle > self.sit_to_stand_hip_angle
                )
                if len(self._frame) > self._frame_len:
                    self._frame = self._frame[-self._frame_len:]
                    
                    if (
                        all(self._frame) and self._reset 
                        and right_body_grad > self.sit_to_stand_body_gradient
                        and left_body_grad > self.sit_to_stand_body_gradient
                    ):
                        self.increment_count()
                        self._reset = False

                    if not any(self._frame):
                        self._reset = True

        return img, self._count
    
    def get_thresh(self):
        return (
            self.sit_to_stand_hip_angle, 
            Util.gradient_to_angle(self.sit_to_stand_body_gradient)
        )


class StepTracker(Generic):
    """
    step-tracking class

    """
    step_tracking_knee_angle = 155
    step_tracking_foot_grad = 0.5
    frame_len = 7

    def __init__(self, is_tracking, left_or_right, ignore_vis=False, debug=False):
        """
        left_or_right: specify left or right side

        """
        super().__init__(is_tracking, ignore_vis, debug)
        self._left_or_right = left_or_right
        self.reset_count()

        self._left_thigh_grad_points = (Motion.left_knee, Motion.left_hip)
        self._right_thigh_grad_points = (Motion.right_knee, Motion.right_hip)

        self._left_knee_angle_points = (Motion.left_ankle, Motion.left_knee, Motion.left_hip)
        self._right_knee_angle_points = (Motion.right_ankle, Motion.right_knee, Motion.right_hip)

        self._left_foot_grad_points = (Motion.left_heel, Motion.left_toes)
        self._right_foot_grad_points = (Motion.right_heel, Motion.right_toes)

        self._vel_frame = list()
        self._vel_frame_len = 5


    def track_movement(self, landmarks, img, source, view):
        """
        track the number of steps taken with the left and right foot respectively

        """
        if view is None:
            return img, self._count

        if view == Util.SIDE_VIEW:
            """
            if camera view is from the side:
            use knee angle and foot slope to track steps

            """
            
            if self._left_or_right == Util.LEFT:
                p_thigh = Util.get_points(landmarks, self._left_thigh_grad_points)
                p_knee = Util.get_points(landmarks, self._left_knee_angle_points)
                p_foot = Util.get_points(landmarks, self._left_foot_grad_points)

            elif self._left_or_right == Util.RIGHT:
                p_thigh = Util.get_points(landmarks, self._right_thigh_grad_points)
                p_knee = Util.get_points(landmarks, self._right_knee_angle_points)
                p_foot = Util.get_points(landmarks, self._right_foot_grad_points)

            else:
                p_thigh, p_knee, p_foot = None, None, None
                
            thigh_grad = self.find_gradient(img.shape, p_thigh, ignore_vis=True)
            knee_angle = self.find_angle(img.shape, p_knee, ignore_vis=True)
            foot_grad = self.find_gradient(img.shape, p_foot, ignore_vis=True)

            if self._debug: 
                img = self.debug(img, source, landmarks, view, knee_angle, foot_grad)

            """ count steps """
            if knee_angle > self.step_tracking_knee_angle and 0 < foot_grad < self.step_tracking_foot_grad and thigh_grad > 1:
                if self._left_or_right == Util.LEFT and self._debug:
                    start = (landmarks[Motion.left_heel][1], landmarks[Motion.left_heel][2])
                    end = (landmarks[Motion.left_toes][1], landmarks[Motion.left_toes][2])
                    cv.line(img, start, end, Util.CYAN, 8)

                elif self._left_or_right == Util.RIGHT and self._debug:
                    start = (landmarks[Motion.right_heel][1], landmarks[Motion.right_heel][2])
                    end = (landmarks[Motion.right_toes][1], landmarks[Motion.right_toes][2])
                    cv.line(img, start, end, Util.CYAN, 8)

                self._frame.append(True)

            elif 0 < knee_angle < self.step_tracking_knee_angle and foot_grad > 0 and thigh_grad > 1:

                self._frame.append(False)

        else:
            """
            if view is not from the side (from the front or back):
            use foot depth information to track steps

            """
            start = (landmarks[Motion.left_hip][1], landmarks[Motion.left_hip][2])
            end = (landmarks[Motion.right_hip][1], landmarks[Motion.right_hip][2])

            if self._debug:
                cv.line(img, start, end, Util.BLUE, 8)

            p_heel = (landmarks[Motion.left_heel][3], landmarks[Motion.right_heel][3])
            p_hip = (landmarks[Motion.left_hip][3], landmarks[Motion.right_hip][3])

            if self._debug:
                img = self.debug(img, source, landmarks, view, p_heel, p_hip)
            
            right_ankle = landmarks[Motion.right_ankle]
            left_ankle = landmarks[Motion.left_ankle]

            cond_vis = [left_ankle[4] > Util.VIS, right_ankle[4] > Util.VIS]

            if self._left_or_right == Util.RIGHT and all(cond_vis):
                if left_ankle[3] > right_ankle[3]:
                    self._frame.append(view == Util.FRONT_VIEW)
                else:
                    self._frame.append(view == Util.REAR_VIEW)

            elif self._left_or_right == Util.LEFT and all(cond_vis):
                if right_ankle[3] > left_ankle[3]:
                    self._frame.append(view == Util.FRONT_VIEW)
                else:
                    self._frame.append(view == Util.REAR_VIEW)
            else:
                self._frame.append(False)

        if len(self._frame) > self.frame_len:
            self._frame = self._frame[-self.frame_len:]

            if not any(self._frame) and self._reset:
                self.increment_count()
                self._reset = False

            if all(self._frame):
                self._reset = True

        return img, self._count
    
    def get_thresh(self):
        return (
            self.step_tracking_knee_angle, 
            Util.gradient_to_angle(self.step_tracking_foot_grad), 
            self.frame_len,
        )
    
    def debug(self, img, source, landmarks, view, *args):

        if view == Util.SIDE_VIEW:

            knee_angle, foot_grad = args

            if self._left_or_right == Util.RIGHT:
                lm_knee, lm_toe = Motion.right_knee, Motion.right_toes
            elif self._left_or_right == Util.LEFT:
                lm_knee, lm_toe = Motion.left_knee, Motion.left_toes
            else:
                lm_knee, lm_heel = None, None

            img = self.annotate(img, source, landmarks, knee_angle, lm=lm_knee, type="angle")
            img = self.annotate(img, source, landmarks, foot_grad, lm=lm_toe, type="grad")

            if self._left_or_right == Util.LEFT:
                ankle_vel = landmarks[Motion.left_ankle][5]
                hip_vel = landmarks[Motion.left_hip][5]
                lm = Motion.left_ankle
            elif self._left_or_right == Util.RIGHT:
                ankle_vel = landmarks[Motion.right_ankle][5]
                hip_vel = landmarks[Motion.right_hip][5]
                lm = Motion.right_ankle

            if ankle_vel is not None and hip_vel is not None:
                self._vel_frame.append(abs(ankle_vel - hip_vel))

            if len(self._vel_frame) > self._vel_frame_len:
                self._vel_frame = self._vel_frame[-self._vel_frame_len:]
                vel = round(mean(self._vel_frame))
                img = self.annotate(img, source, landmarks, vel, lm=lm, type="vel")

        else:
            p_heel, p_hip = args
            img = self.annotate(img, source, landmarks, p_heel[0], lm=Motion.left_heel, type="grad")
            img = self.annotate(img, source, landmarks, p_heel[1], lm=Motion.left_hip, type="grad")
            img = self.annotate(img, source, landmarks, p_hip[0], lm=Motion.right_heel, type="grad")
            img = self.annotate(img, source, landmarks, p_hip[1], lm=Motion.right_hip, type="grad")

        return img


class StandingTimer(Generic):
    """
    class to track stanting time

    """
    standing_timer_hip_angle = 150
    standing_timer_body_gradient = 5

    def __init__(self, is_tracking, ignore_vis=False, debug=False):

        super().__init__(is_tracking, ignore_vis, debug)
        self.reset_time()

    def reset_time(self):
        """
        resets standing timer

        """
        self._start_time = None
        self._non_standing_time = 0
        self._prev_time = None

        self._left_hip_angle_points = (Motion.left_knee, Motion.left_hip, Motion.left_shoulder)
        self._right_hip_angle_points = (Motion.right_knee, Motion.right_hip, Motion.right_shoulder)

    def track_movement(self, img, landmarks):
        """
        get the total standing time since the start of the session

        """
        is_standing = False

        if self._start_time is None and self._is_tracking:
            self._start_time = time.time()

        if len(landmarks) > 0:

            p = Util.get_points(landmarks, self._left_hip_angle_points)
            left_hip_angle = self.find_angle(img.shape, p, ignore_vis=True)

            p = Util.get_points(landmarks, self._right_hip_angle_points)
            right_hip_angle = self.find_angle(img.shape, p, ignore_vis=True)

            if (
                (left_hip_angle > self.standing_timer_hip_angle or right_hip_angle > self.standing_timer_hip_angle)
                and True ### replace with body gradient check
            ):
                is_standing = True

        diff_time = time.time() - self._prev_time if self._prev_time is not None else 0
        if not is_standing or diff_time > 1:
            self._non_standing_time += diff_time

        self._prev_time = time.time()

        return int(time.time() - self._start_time - self._non_standing_time)

    def get_thresh(self):
        return (
            self.standing_timer_hip_angle, 
            Util.gradient_to_angle(self.standing_timer_body_gradient)
        )
    

class BoxAndBlocks(Generic):
    """
    box and blocks hand-tracking class

    """

    def __init__(self, is_tracking, left_or_right, ignore_vis=False, debug=False):
        """
        left_or_right: whether to track the left or right hand

        """
        super().__init__(is_tracking, ignore_vis, debug)
        self._left_or_right = left_or_right

        self.reset_count()

    def reset_count(self):
        """
        init count to zero by default

        """
        self._count = 0

        self._prev = None
        self._curr = None

        self._prev_handedness = None

    def track_movement(self, hand_landmarks, boundary, handedness):
        """
        increments hand counts for left and right hands respectively based on
        detected hand positions relative to the detected boundary

        """

        """ don't count if no boundary is detected """
        if boundary is None:
            return self._count

        """ attempt to count if hand is detected """
        if len(hand_landmarks) != 0:
            x, y = hand_landmarks[Hand.pinky][1:3]

            """ check for left or right handedness """
            if self._left_or_right == Util.RIGHT:
                cond = [handedness == Util.RIGHT, self._prev_handedness == Util.RIGHT]
            elif self._left_or_right == Util.LEFT:
                cond = [handedness == Util.LEFT, self._prev_handedness == Util.LEFT]

            if all(cond) and self._left_or_right == Util.RIGHT:
                self._curr = True if x > boundary[Util.X] else False
            elif all(cond) and self._left_or_right == Util.LEFT:
                self._curr = True if x < boundary[Util.X] else False

            """ count based on pinky position relative to the detected boundary """
            if self._prev == False and self._curr == True:
                if all(cond) and y < boundary[Util.Y]:
                    self.increment_count()

            """ use previous handedness information if unsure of current """
            if handedness is not None:
                self._prev_handedness = handedness

            self._prev = self._curr

        return self._count


class BoundaryDetector:
    """
    generic boundary detector

    """

    lst_size = 10

    def __init__(self, aruco=True):
        """
        aruco: a boolean to specify whether or not aruco markers are being used for boundary detection

        """
        self._aruco = aruco

        """ arucos """
        self._x = list()
        self._y = list()

        self._prev_x = None
        self._prev_y = None

        self._point = None

    def detect_boundary(self, img, detected, corr_mode=False):
        """
        attempt to detect the boundary in the current frame

        """
        self._point = self._point if corr_mode else None

        def get_avg(data):
            total, count = 0, 0
            for d in data:
                if d is not None:
                    total += d
                    count += 1

            return total // count if count != 0 else None

        """ 
        use aruco position if specified 
        otherwise manually detect boundary by looking at prominent vertical line in the frame

        """
        h, w, _ = img.shape

        if len(detected) > 0:
            id, x, y = detected[0]

            '''
            if self._prev_x is not None and self._prev_y is not None:

                if x < self._prev_x - 0.1*w or x > self._prev_x + 0.1*w:
                    self._x.append(self._prev_x)
        
                if y < self._prev_y - 0.1*h or y > self._prev_y + 0.1*h:
                    self._y.append(self._prev_y)

            else:
                self._x.append(x)
                self._y.append(y)
            '''
            
            self._x.append(x)
            self._y.append(y)
                
        else:
            self._x.append(None)
            self._y.append(None)

        if len(self._x) > self.lst_size:
            self._x = self._x[-self.lst_size:]
            x_avg = get_avg(self._x)
        else:
            x_avg = None

        if len(self._y) > self.lst_size:
            self._y = self._y[-self.lst_size:]
            y_avg = get_avg(self._y)
        else:
            y_avg = None

        if x_avg is not None and y_avg is not None:
            self._point = (x_avg, y_avg)
            cv.circle(img, self._point, 12, Util.RED, 5)

        self._prev_x, self._prev_y = x_avg, y_avg

        return img, self._point
