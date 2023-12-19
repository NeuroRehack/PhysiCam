"""
main.py

see "doc/main.md" for more details

"""

import sys
from PyQt5 import QtWidgets
from window import MainWindow


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "12/04/2023"
__status__ = "Prototype"
__credits__ = ["Agnethe Kaasen", "Live Myklebust", "Amber Spurway"]


def main():
    print(f"Python Version Info: {sys.version}")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__": 
    main()
