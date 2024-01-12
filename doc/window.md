# Main Window File

- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 12/04/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

- `class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow)`
    - Front-end thread
    - Handles user input to the graphical user interface
    - Inherits from `QtWidgets.QMainWindow`, `QtWidgets.QWidget` and `gui_main.Ui_MainWindow`

## Main Mindow Thread methods

`def __init__(self, parent=None)`
- Sets up graphical user interface
- Creates an instance of the main-worker thread.
- Connects the following signals:
    - All back-end signals
    - Motion tracking signals
    - Pushbutton signals
    - Line-edit signals
    - Action menu triggers
- Init and connect button controls:
    - Start / Stop pushbutton
    - Pause / Resume pushbutton

`def closeEvent(self, event)`
- Callback for when the user exit the program
- `event`: the event that triggered the callback

`def update_frame(self, img)`
- Updates GUI interface whenever a new video frame is received from the main worker thread
- `img`: object containing the current video frame, emitted by the main-worker thread

`def display_frame_rate(self, frame_rate)`
- Shows the current frame rate on the gui 
- Takes the average of the last ten frame rates for smoother output
- `frame_rate`: the current calculated frame rate emitted from the main-worker thread

`def display_session_time(self, time)`
- Displays the time since the start of session formatted as "h:mm:ss"
- `time`: the elapsed time since the start of the session

`def update_start_pushButton(self)`
- Updates the gui interface whenever the start / stop button is pressed

`def update_name_id(self)`
- Called whenever the user enters anything into the name line edit
- Passes this information to the main-worker thread

`def open_file(self)`
- Callback for when the open action is triggered from the file menu
- Gets the file name / path and sends to the main-worker thread

`def open_webcam(self)`
- Callback for when the webcam action is triggered from the file menu
- Starts the video capture from the webcam in the main-worker thread

`def generate_file(self)`
- Callback for when the generate csv file action is triggeres from the file menu
- Passes the current "generate file" status to the main-worker thread