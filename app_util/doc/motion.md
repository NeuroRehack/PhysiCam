# Motion Tracking Module
- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 12/04/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

Motion tracking module. Handles the passing of information between the "MediaPipe Pose Estimation" library and the main application.

The "MediaPipe Pose Estimation" library was chosen for this application because of the following key benefits:
- ML models are lightweight, does not require high performance computers to run.
- Runs in real-time, has low latency and is able to maintain a reasonable frame rate (>10fps).
- Leverages advanced ML models to achieve high accuracy in pose estimation.
- Provides a modular and flexible architecture that allows for customisation. Can set confidence level thresholds, choose model complexity etc.
- Can handle challenging environments, ensuring robust pose estimation. Can handle scenarios where body parts are partially or fully obstructed and still provide reasonably accurate results.

Specifically for this project, motion tracking will require a 90% high confidence level for detection. Continuous tracking will require a 50% confidence level. This is to ensure the program is sure of the subject prior to tracking, but stay locked on to the subject while tracking.

Motion tracking confidence levels:
```
min_detection_confidence = 0.9
min_tracking_confidence = 0.5
```

Motion tracking landmark ID's:
```
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
```

Other motion tracking landmark ID's (not used in this project):
```
nose = 0

left_eye_inner = 1
left_eye = 2
left_eye_outer = 3

right_eye_inner = 4
right_eye = 5
right_eye_outer = 6

left_ear = 7
right_ear = 8

mouth_left = 9
mouth_right = 10

left_pinky = 17
right_pinky = 18

left_index = 19
right_index = 20

left_thumb = 21
right_thumb = 22

left_heel = 29
right_heel = 30

left_foot = 31
right_foot = 32
```

## Module methods

`def __init__(self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)`
- Initialises "MediaPipe Pose Estimation" for motion tracking.
- Sets the following default values:
    - `static_image_mode`: 
        - Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream.
        - Set to `False` by default to allow for more accurate motion tracking in video streams.
    - `model_complexity`: 
        - Complexity of the pose landmark model: 0, 1 or 2.
        - Set to `1` by defualt to provide a balance between accuracy and performace.
    - `smooth_landmarks`:
        - Whether to filter landmarks across different input images to reduce jitter.
        - Set to `True` by defualt to allow for smoother landmark values and to reduce the effect of noise on the landmark values.
    - `enable_segmentation`: 
        - Whether to predict segmentation mask.
        - Set to `False` by default as it is not used for this application.
    - `smooth_segmentation`: 
        - Whether to filter segmentation across different input images to reduce jitter.
        - Set to `True` by default to allow for smoother subject detection.
    - `min_detection_confidence`: 
        - Minimum confidence value (between 0 and 1) for person detection to be considered successful.
        - Set to the `min_detection_confidence` value mentioned in "Description".
    - `min_tracking_confidence`: 
        - Minimum confidence value (between 0 and 1) for the pose landmarks to be considered tracked successfully.
        - Set to the `min_tracking_confidence` value mentioned "Description".
- source: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose.py

`def track_motion(self, img, landmarks)`
- Used for tracking motion within a bounding box
- `img`: Current video frame
- `landmarks`: List of co-ordinate values of all detected landmarks.
- Crops the frame based on the position of the detected person in the previous frame
- Looks for human motion in the bounding box / cropped frame
- If human motion is detected, overlay the stick figure on the frame
- Calculates co-ordinate values in pixels to be used later for drawing the bounding box and to crop the next frame
- Highlight important points (wrists, elbows, shoulders, hips, knees, ankles)
- Draws connections between detected points
- Draws the bounding box
- Returns the current video frame with the detected elements back-projected onto the frame and a copy of the landmark co-ordinated in pixel values.
- **Tech Requirement 1.2:** Usability, User Interface:
    - The video display must have all tracking points and appropriate line connections (eg: arm and leg connections) displayed onto the video while recording is active.
    - The video display must show a bounding box around the current tracking subject if recording is active. Only one subject is to be tracked at any given time.
- **Tech Requirement 2.4:** Data Capturing, Data Precision: 
    - The motion tracking should at least track the following points on the human body on the left and right side (wrists, elbows, shoulders, hips, knees and ankles)
- **Tech Requirement 6.10:** Performance, Device Independance:
    - The program must be able to run on all computers with Windows 10 or later