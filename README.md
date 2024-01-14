# PhysiCam

## Project
- Quantifying physical therapy with an easy-to-use, intuitive tool

## Description & Features
- A motion tracking application for counting reps during a physiotherapy session.
- Uses computer webcam by default, able to use an external USB webcam. Supports simultaneous use of multiple cameras connected to host device.
- Tracks motion using the "MediaPipe Pose Estimation" library and counts movements in real-time. Can be used with Coral Edge TPU (requires Python v3.8, disabled in exe file due to compatibility).
- Saves recorded session information to a csv file under patient name or ID number. Video recording (disabled by default) can also be enabled.
- Able to apply motion tracking and counting reps on mp4 videos.

## Requirements
- Works on Windows 10 or later
- Works with Python version v3.8 or later (Python v3.8 is required if used with Coral Edge TPU)

## Setup

<!--
### Mac / Linux
1.  Create python virtual environment: `python3 -m venv venv`
2.  Activate virtual environment: `source venv/bin/activate`
3.  Install dependencies: `python3 -m pip install -r req.txt`
4.  Run the program: `python3 src/main.py`
5.  To deactivate virtual environment: `deactivate`
-->

#### Windows
1.  Create python virtual environment: `py -m venv venv`
2.  Activate virtual environment: `.\\venv\Scripts\activate`
3.  Install dependencies: `py -m pip install -r req.txt`
4.  Run the program: `py main.py`
5.  To deactivate virtual environment: `deactivate`

## Graphical User Interface
- Generate GUI File: `pyuic5 -x ui/gui.ui -o gui.py`

## Genetate EXE File
```
pyinstaller main.py ^
    --add-data=C:\Users\61416\AppData\Local\Programs\Python\Python310\Lib\site-packages\mediapipe\modules;mediapipe/modules ^
    --add-data="C:\Users\61416\Documents\Uni_5\PhysiCam\app_util\images\icon.png";app_util/images/icon.png ^
    --icon="C:\Users\61416\Documents\Uni_5\PhysiCam\app_util\images\icon.png" ^
    --onefile -w
```