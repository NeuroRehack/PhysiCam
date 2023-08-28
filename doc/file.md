# File Module
- Author: Mike Smith
- Email: dongming.shi@uqconnect.edu.au
- Date of Implementation: 01/05/2023
- Status: Prototype
- Credits: Agnethe Kaasen, Live Myklebust, Amber Spurway

## Description

Handles reading and writing files. Contains methods for checking invalid files, parsing movement data and writing to csv files.

Supported files:
- `.csv`
- `.mp4`

## Module methods

`def __init__(self, save=True)`
- `save`: a boolean to specify whether or not to generate a file
- **Tech Requirement 2.3:** Data Capturing, Data Storage: The program must allow user to disable generating csv files to store session data.

`def set_save_status(self, save)`
- Set whether or not to generate a file
- `save`: a boolean to specify whether or not to generate a file

`def get_file_type(self, filename)`
- Checks that the file is supported by the program (eg: .mp4 videos or .csv files)
- Returns the file type
- `filename`: the name of the file to be opened
- **Tech Requirement 2.5:** Data Capturing, Video Playback: The video must be and .mp4 video.

`def read(self)`
- Not implemented yet

`def write(self)`
- Takes the parsed data and writes it to a csv file
- Adds patient name or ID to filename is specified
- **Tech Requirement 2.3:** Data Capturing, Data Storage: 
    - The program must generate a .csv file containing information regarding the patients, the number of reps and coordinates of body parts with time stamps. 
    - The outputted .csv file must contain patient name or ID number if information is provided to the program.
- **Tech Requirement 4.6:** Privacy, Data Security: 
    - Raw footage of the recorded session must not be saved in any way on the local device.

`def create_filename(self)`
- Creates a unique filename using the current system time and date
- Returns a unique filename
- **Tech Requirement 2.3:** Data Capturing, Data Storage: All outputted .csv filenames contain the current time and data when the recording session is stopped.

`def parse_movements(self, movements, landmarks, curr_time)`
- Parses the movement data periodically and stores it to be written later
- `movements`: dictionary of movements containing tracking status
- `landmarks`: list containing all motion tracking landmark co-ordinates
- `curr_time`: time elapsed since start of the session in seconds
- **Tech Requirement 2.3:** Data Capturing, Data Storage:
    - The generated .csv files save the number of reps for each movement and the co-ordinate values every 100ms and is timestamped
    - The stored co-ordinate values are numbers between 0 and 1 representing the position of the point on the frame. (0, 0) represents top-left and (1, 1) represents (bottom right). Invalid co-ordinates are represented by a number < 0 or > 1.