# Movement Module
- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 23/04/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

Contains a generic movement class. Each movement is defined by a set of angle (each defined by three points) and a set of positional thresholds (each define by two points).

## Module methods

`def __init__(self, points, positions, is_tracking, ignore_vis=False, debug=False)`
- `points`: a list containing tuples of three points and the threshold angle
- `positions`: a list containing pairs of point with thresholds for relative positioning
- `is_tracking`: a boolean to specify whether tracking is enabled
- `ignore_vis`: a boolean to ignore visibility thresholds set to "True" if tracking full-body / compound movements
- `debug`: a boolean to allow program to overlay angle values on frame

`def invalid_num_of_elements_err(self, i)`
- Checks for invalid lengths in the input arrays.
- `i`: the current index to be checked.
- Raises a `ValueError` is invalid number of elements is detected.

`def reset_count(self)`
- Initialise count to zero by default

`def set_tracking_status(self, is_tracking)`
- Can be used to change the tracking status of each movement dynamically.
- `is_tracking`: boolean to specify whether tracking is enabled for a specific movement.

`def get_tracking_status(self)`
- Gets the current tracking status.
- Return the tracking status.

`def count_movement(self, landmarks, pixels, img, source)`
- Count the number of reps for the movement
- The count value is only incremented if all the angular and positional threasholds are met. Once a rep is counted, the movement goes into the "set" state after which the next rep is only counted once the movement returns to the "reset" state.
- If all angular and positional threasholds are not satisfied, the movement enters the "reset" state where the process repeats.
- `landmarks`: a list of positional values for all detected landmarks.
- `pixels`: a list of pixel co-ordinated for all detected landmarks
- `img`: the current video frame
- `source`: the current video source: (video or webcam)
- Returns the current video frame and the current movement count.
- **Tech Requirement 2.4:** Data Capturing, Data Precision:
    - The motion tracking software is able to track arm extensions and sit to stand using the appropriate co-ordinate values.

`def get_count(self)`
- Returns the current movement count value

`def find_angle(self, p1, p2, p3)`
- Generic function for calutating the angle between two lines defined by three points `p1`, `p2`, and `p3`
- Note: `p2` must be the common point between the two lines
- Checks that the visibility of the points is above the visibility threshold. Otherwise, ignore points as this can lead to program guessing position of points.
- Checks that points are not too close to the edge of the frame. Otherwise, ignore points as this can lead to inaccurate angle values.
- Make sure all angle values are between 0 and 180 degrees
- Returns the final calculated angle value

`def get_position(self, pos, landmarks, x_or_y)`
- Returns the current x or y position of specifies landmark
- `pos`: the index position of the specified landmark (same ad the landmark id)
- `landmarks`: a list of all tracking landmarks
- `x_or_y`: whether to get the x or y co-ordinate of the given point.

`def annotate(self, img, source, pixels, angle, index)`
- Overlays angle values onto video frames
- `img`: the current video frame
- `source`: the current video source: (video or webcam)
- `pixels`: a list of pixel co-ordinated for all detected landmarks
- `angle`: the angle value to be annotated onto the frame.
- `index`: the index of the current angle to annotate
- Returns the current video frame with the annotated angles