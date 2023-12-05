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

`DEFAULT_FILE_PATH`: Default path for csv files to be saved: "./files"

`FILE_NOT_SUPPORTED`: Invalid file: -1

`CSV`: .csv file: 0

`MP4`: .mp4 video: 1

### Maximum Frame Dimensions (Full-HD)

`FRAME_WIDTH`: Max width of the video frame: 1920

`FRAME_HEIGHT`: Max height of the video frame: 1080

### Positional Thresholds (normalised pixel co-ordinates)

Landmark co-ordinates that are close to the edges of the frame often return inaccurate values. Therefore points that lay outside of these tresholds (near the edges of the frame) will be ignored by the "Movement Module" when counting reps.

`MIN`: Minimum positional threshold: 0.02

`MAX`: Maximum positional threshold: 0.98

### Visibility Threshold (between 0 and 1)

All detected landmarks have a visibility value. Landmark points with a low visibility values often return inaccurate co-ordinate values as a result of the "Pose Estimation" predicting low visibility points. Therefore points that have a visibility value less than the threshold are ignored by the "Movement Module" when counting reps.

`VIS`: Minimum visibility threshold: 0.5

### Input Source Definitions

`VIDEO`: Input source from .mp4 video: 0

`WEBCAM`: Input source from webcam: 1

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

#### Co-ordinates
```
X = 0
Y = 1
```
