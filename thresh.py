"""
thresh.py

see "doc/thresh.md" for more details

"""

from PyQt5 import QtCore, QtWidgets
from app_util import gui_thresh
from app_util.util import Util


__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "19/12/2023"
__status__ = "Prototype"
__credits__ = ["Agnethe Kaasen", "Live Myklebust", "Amber Spurway"]


class ThreshWindow(QtWidgets.QMainWindow, QtWidgets.QWidget, gui_thresh.Ui_MainWindow):
    """
    Threshold Window: for the user to adjust various thresholds related to counting reps
    (in development)

    """
    sit_to_stand_hip_angle = QtCore.pyqtSignal(int)

    left_arm_reach_elbow_angle = QtCore.pyqtSignal(int)
    left_arm_reach_shoulder_angle = QtCore.pyqtSignal(int)

    right_arm_reach_elbow_angle = QtCore.pyqtSignal(int)
    right_arm_reach_shoulder_angle = QtCore.pyqtSignal(int)

    def __init__(self, tracking_movements, parent=None):
        super().__init__(parent)

        """ set up gui """
        self.setupUi(self)
        self.setWindowTitle("PhysiCam - Adjust Thresholds")
        self.setWindowIcon(Util.get_icon())

        self._tracking_movements = tracking_movements

        self.sitToStand_hipAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.sit_to_stand_hip_angle.emit(int(value))
        )
        self.leftArmReach_elbowAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.left_arm_reach_elbow_angle.emit(int(value))
        )
        self.leftArmReach_shoulderAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.left_arm_reach_shoulder_angle.emit(int(value))
        )
        self.rightArmReach_elbowAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.right_arm_reach_elbow_angle.emit(int(value))
        )
        self.rightArmReach_shoulderAngle_horizontalSlider.valueChanged.connect(
            lambda value: self.right_arm_reach_shoulder_angle.emit(int(value))
        )

        value = self._tracking_movements.get(Util.RIGHT_ARM_REACH).get_thresh()[0]
        self.rightArmReach_elbowAngle_horizontalSlider.setValue(value)
        self.right_elbow_label.setText(f"{value}")

        value = self._tracking_movements.get(Util.RIGHT_ARM_REACH).get_thresh()[1]
        self.rightArmReach_shoulderAngle_horizontalSlider.setValue(value)
        self.right_shoulder_label.setText(f"{value}")

        value = self._tracking_movements.get(Util.LEFT_ARM_REACH).get_thresh()[0]
        self.leftArmReach_elbowAngle_horizontalSlider.setValue(value)
        self.left_elbow_label.setText(f"{value}")

        value = self._tracking_movements.get(Util.LEFT_ARM_REACH).get_thresh()[1]
        self.leftArmReach_shoulderAngle_horizontalSlider.setValue(value)
        self.left_shoulder_label.setText(f"{value}")