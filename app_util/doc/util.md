# Utilities File
- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 22/04/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

Contains static definitions specifically used for this project.

## Definitions

### File Related Definitions

`DEFAULT_FILE_PATH`: Default path for csv files to be saved: "./PhysiCam_RecordedSessions"
`DEFAULT_VIDEO_PATH`: Default path for video files to be saved: "./Physicam_VideoRecordings"

`FILE_SAVE_INTERVAL`: Amount of time elapsed before saving csv for video file. Used for saving files periodically.

`FILE_NOT_SUPPORTED`: Invalid file: -1
`CSV`: .csv file: 0
`MP4`: .mp4 video: 1
`AVI`: .avi video: 2

### Maximum Frame Dimensions (Full-HD)

`FRAME_WIDTH`: Max width of the video frame: 1920
`FRAME_HEIGHT`: Max height of the video frame: 1080
`FRAME_ORIGIN`: Origin (zero point) of the frame: 0

### Positional Thresholds (normalised pixel co-ordinates)

Landmark co-ordinates that are close to the edges of the frame often return inaccurate values. Therefore points that lay outside of these tresholds (near the edges of the frame) will be ignored by the "Movement Module" when counting reps.

`MIN`: Minimum positional threshold: 0.01
`MAX`: Maximum positional threshold: 0.99

### Visibility Threshold (between 0 and 100)

All detected landmarks have a visibility value. Landmark points with a low visibility values often return inaccurate co-ordinate values as a result of the "Pose Estimation" predicting low visibility points. Therefore points that have a visibility value less than the threshold are ignored by the "Movement Module" when counting reps.

`VIS`: Minimum visibility threshold: 50

### Input Source Definitions

`VIDEO`: Input source from .mp4 video: 0
`WEBCAM`: Input source from webcam: 1

### Multiple Cameras

`MAX_NUM_CAMERAS`: Max number of cameras supported by program: 64
`MULTI_CAM_DELAY_THRESH`: Delay threshold for counting using multiple cameras: 0.5 seconds

### Other Definitions

#### Colours

```
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
```

#### Camera Angle Definitions (used for steps tracking)
```
SIDE_VIEW = 0
FRONT_VIEW = 1
REAR_VIEW = 2
```

#### Co-ordinates
```
X = 0
Y = 1
```

#### Left or Right
```
LEFT = "Left"
RIGHT = "Right"
```

#### Movement Labels
```
RIGHT_ARM_REACH = "Right Arm Reach"
LEFT_ARM_REACH = "Left Arm Reach"
SIT_TO_STAND = "Sit to Stand"
RIGHT_STEPS = "Right Steps"
LEFT_STEPS = "Left Steps"
STANDING_TIME = "Standing Time"
LEFT_HAND = "Left Hand"
RIGHT_HAND = "Right Hand"
```

### Functions

`def get_icon()`: loads the PhysiCam icon

`def get_points()`: extracts indexed points from an array

`def create_filename()`: creates a unique timestamped filename that is shared among csv files and videos.

`def low_pass_filter()`: function for creating a lpf.

`def gradient_to_angle()`: converts a gradient value to angle (with respect to the horizontal)

`def angle_to_gradient()`: converts an angle value (with respect to the horizontal) to a gradient value.