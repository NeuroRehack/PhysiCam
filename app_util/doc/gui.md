# GUI Module
- Form implementation generated from reading ui file 'ui/gui.ui'
- Created by: PyQt5 UI code generator 5.15.9
- Date of Implementation: 16/04/2023

## Description:

Sets the layout of the graphical user interface. Does not handle any logic or user inputs. Only provides variables that reference respective on-screen elements. 

This module is auto-generated from the "ui/gui.ui" file. The following command can be used to generate this module:

`pyuic5 -x ui/gui.ui -o gui.py`

## Module methods:

`def setupUi(self, MainWindow)`
- Sets up the layout of the graphical user interface.
- **Tech Requirement 1.2:** Usability, User Interface:
    - The user interface should provide relevant information such as session time, frame rate, type of movement and counting.
    - The labels for counting movement repititions should be viewable from a 5 metre distance.
    - The application must contain a text box to allow the user to input a patient name or ID number to allow for unique identification.

`def retranslateUi(self, MainWindow)`
- Set up defaults for the user interface elements. Eg: Sets up default text for all labels and action menu elements.