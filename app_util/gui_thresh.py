# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/gui_thresh.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 1186, 750))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.scrollAreaWidgetContents_3)
        self.frame.setMinimumSize(QtCore.QSize(700, 500))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.sitToStand_hipAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_4)
        self.sitToStand_hipAngle_horizontalSlider.setMinimum(90)
        self.sitToStand_hipAngle_horizontalSlider.setMaximum(180)
        self.sitToStand_hipAngle_horizontalSlider.setSingleStep(1)
        self.sitToStand_hipAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sitToStand_hipAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sitToStand_hipAngle_horizontalSlider.setTickInterval(0)
        self.sitToStand_hipAngle_horizontalSlider.setObjectName("sitToStand_hipAngle_horizontalSlider")
        self.gridLayout_3.addWidget(self.sitToStand_hipAngle_horizontalSlider, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.sitToStand_hip_label = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.sitToStand_hip_label.setFont(font)
        self.sitToStand_hip_label.setObjectName("sitToStand_hip_label")
        self.gridLayout_3.addWidget(self.sitToStand_hip_label, 0, 2, 1, 1)
        self.sitToStand_body_label = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.sitToStand_body_label.setFont(font)
        self.sitToStand_body_label.setObjectName("sitToStand_body_label")
        self.gridLayout_3.addWidget(self.sitToStand_body_label, 1, 2, 1, 1)
        self.sitToStand_bodyAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_4)
        self.sitToStand_bodyAngle_horizontalSlider.setMinimum(0)
        self.sitToStand_bodyAngle_horizontalSlider.setMaximum(85)
        self.sitToStand_bodyAngle_horizontalSlider.setSingleStep(1)
        self.sitToStand_bodyAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sitToStand_bodyAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sitToStand_bodyAngle_horizontalSlider.setTickInterval(0)
        self.sitToStand_bodyAngle_horizontalSlider.setObjectName("sitToStand_bodyAngle_horizontalSlider")
        self.gridLayout_3.addWidget(self.sitToStand_bodyAngle_horizontalSlider, 1, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.leftArmReach_shoulderAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_5)
        self.leftArmReach_shoulderAngle_horizontalSlider.setMaximum(90)
        self.leftArmReach_shoulderAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.leftArmReach_shoulderAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.leftArmReach_shoulderAngle_horizontalSlider.setObjectName("leftArmReach_shoulderAngle_horizontalSlider")
        self.gridLayout_4.addWidget(self.leftArmReach_shoulderAngle_horizontalSlider, 4, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 4, 0, 1, 1)
        self.leftArmReach_elbowAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_5)
        self.leftArmReach_elbowAngle_horizontalSlider.setMinimum(90)
        self.leftArmReach_elbowAngle_horizontalSlider.setMaximum(180)
        self.leftArmReach_elbowAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.leftArmReach_elbowAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.leftArmReach_elbowAngle_horizontalSlider.setObjectName("leftArmReach_elbowAngle_horizontalSlider")
        self.gridLayout_4.addWidget(self.leftArmReach_elbowAngle_horizontalSlider, 2, 1, 1, 1)
        self.left_elbow_label = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.left_elbow_label.setFont(font)
        self.left_elbow_label.setObjectName("left_elbow_label")
        self.gridLayout_4.addWidget(self.left_elbow_label, 2, 2, 1, 1)
        self.left_shoulder_label = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.left_shoulder_label.setFont(font)
        self.left_shoulder_label.setObjectName("left_shoulder_label")
        self.gridLayout_4.addWidget(self.left_shoulder_label, 4, 2, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.rightArmReach_shoulderAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_6)
        self.rightArmReach_shoulderAngle_horizontalSlider.setMinimum(0)
        self.rightArmReach_shoulderAngle_horizontalSlider.setMaximum(90)
        self.rightArmReach_shoulderAngle_horizontalSlider.setProperty("value", 0)
        self.rightArmReach_shoulderAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rightArmReach_shoulderAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rightArmReach_shoulderAngle_horizontalSlider.setObjectName("rightArmReach_shoulderAngle_horizontalSlider")
        self.gridLayout_5.addWidget(self.rightArmReach_shoulderAngle_horizontalSlider, 1, 1, 1, 1)
        self.rightArmReach_elbowAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_6)
        self.rightArmReach_elbowAngle_horizontalSlider.setMinimum(90)
        self.rightArmReach_elbowAngle_horizontalSlider.setMaximum(180)
        self.rightArmReach_elbowAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rightArmReach_elbowAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rightArmReach_elbowAngle_horizontalSlider.setObjectName("rightArmReach_elbowAngle_horizontalSlider")
        self.gridLayout_5.addWidget(self.rightArmReach_elbowAngle_horizontalSlider, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 0, 0, 1, 1)
        self.right_elbow_label = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.right_elbow_label.setFont(font)
        self.right_elbow_label.setObjectName("right_elbow_label")
        self.gridLayout_5.addWidget(self.right_elbow_label, 0, 2, 1, 1)
        self.right_shoulder_label = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.right_shoulder_label.setFont(font)
        self.right_shoulder_label.setObjectName("right_shoulder_label")
        self.gridLayout_5.addWidget(self.right_shoulder_label, 1, 2, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_6)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_9 = QtWidgets.QLabel(self.groupBox_7)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout_6.addWidget(self.label_9, 0, 0, 1, 1)
        self.standingTimer_hipAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_7)
        self.standingTimer_hipAngle_horizontalSlider.setMinimum(90)
        self.standingTimer_hipAngle_horizontalSlider.setMaximum(180)
        self.standingTimer_hipAngle_horizontalSlider.setSingleStep(1)
        self.standingTimer_hipAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.standingTimer_hipAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.standingTimer_hipAngle_horizontalSlider.setTickInterval(0)
        self.standingTimer_hipAngle_horizontalSlider.setObjectName("standingTimer_hipAngle_horizontalSlider")
        self.gridLayout_6.addWidget(self.standingTimer_hipAngle_horizontalSlider, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_7)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 1, 0, 1, 1)
        self.standingTimer_bodyAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_7)
        self.standingTimer_bodyAngle_horizontalSlider.setMinimum(0)
        self.standingTimer_bodyAngle_horizontalSlider.setMaximum(85)
        self.standingTimer_bodyAngle_horizontalSlider.setSingleStep(1)
        self.standingTimer_bodyAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.standingTimer_bodyAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.standingTimer_bodyAngle_horizontalSlider.setTickInterval(0)
        self.standingTimer_bodyAngle_horizontalSlider.setObjectName("standingTimer_bodyAngle_horizontalSlider")
        self.gridLayout_6.addWidget(self.standingTimer_bodyAngle_horizontalSlider, 1, 1, 1, 1)
        self.standing_hip_label = QtWidgets.QLabel(self.groupBox_7)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.standing_hip_label.setFont(font)
        self.standing_hip_label.setObjectName("standing_hip_label")
        self.gridLayout_6.addWidget(self.standing_hip_label, 0, 2, 1, 1)
        self.standing_body_label = QtWidgets.QLabel(self.groupBox_7)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.standing_body_label.setFont(font)
        self.standing_body_label.setObjectName("standing_body_label")
        self.gridLayout_6.addWidget(self.standing_body_label, 1, 2, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_7)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_10 = QtWidgets.QLabel(self.groupBox_8)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_7.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_8)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 1, 0, 1, 1)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_8)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setMinimum(90)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setMaximum(180)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setSingleStep(1)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setTickInterval(0)
        self.stepsTracking_sideView_kneeAngle_horizontalSlider.setObjectName("stepsTracking_sideView_kneeAngle_horizontalSlider")
        self.gridLayout_7.addWidget(self.stepsTracking_sideView_kneeAngle_horizontalSlider, 0, 1, 1, 1)
        self.stepsTracking_sideView_footAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_8)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setMinimum(0)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setMaximum(85)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setSingleStep(1)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setTickInterval(0)
        self.stepsTracking_sideView_footAngle_horizontalSlider.setObjectName("stepsTracking_sideView_footAngle_horizontalSlider")
        self.gridLayout_7.addWidget(self.stepsTracking_sideView_footAngle_horizontalSlider, 1, 1, 1, 1)
        self.steps_side_knee_label = QtWidgets.QLabel(self.groupBox_8)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.steps_side_knee_label.setFont(font)
        self.steps_side_knee_label.setObjectName("steps_side_knee_label")
        self.gridLayout_7.addWidget(self.steps_side_knee_label, 0, 2, 1, 1)
        self.steps_side_foot_label = QtWidgets.QLabel(self.groupBox_8)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.steps_side_foot_label.setFont(font)
        self.steps_side_foot_label.setObjectName("steps_side_foot_label")
        self.gridLayout_7.addWidget(self.steps_side_foot_label, 1, 2, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_8)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_9.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_9.setFont(font)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider = QtWidgets.QSlider(self.groupBox_9)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setMinimum(1)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setMaximum(10)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setSingleStep(1)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setTickInterval(0)
        self.stepCounter_frontOrRear_sensitivity_horizontalSlider.setObjectName("stepCounter_frontOrRear_sensitivity_horizontalSlider")
        self.gridLayout_8.addWidget(self.stepCounter_frontOrRear_sensitivity_horizontalSlider, 0, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_9)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_8.addWidget(self.label_11, 0, 0, 1, 1)
        self.steps_front_rear_sensitivity_label = QtWidgets.QLabel(self.groupBox_9)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.steps_front_rear_sensitivity_label.setFont(font)
        self.steps_front_rear_sensitivity_label.setObjectName("steps_front_rear_sensitivity_label")
        self.gridLayout_8.addWidget(self.steps_front_rear_sensitivity_label, 0, 2, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_9)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 2)
        self.groupBox_12 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_12.setMinimumSize(QtCore.QSize(360, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_12.setFont(font)
        self.groupBox_12.setObjectName("groupBox_12")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_12)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_23 = QtWidgets.QLabel(self.groupBox_12)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.gridLayout_11.addWidget(self.label_23, 0, 0, 1, 1)
        self.multipleCams_delay_horizontalSlider = QtWidgets.QSlider(self.groupBox_12)
        self.multipleCams_delay_horizontalSlider.setMinimum(0)
        self.multipleCams_delay_horizontalSlider.setMaximum(2000)
        self.multipleCams_delay_horizontalSlider.setSingleStep(100)
        self.multipleCams_delay_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.multipleCams_delay_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.multipleCams_delay_horizontalSlider.setTickInterval(0)
        self.multipleCams_delay_horizontalSlider.setObjectName("multipleCams_delay_horizontalSlider")
        self.gridLayout_11.addWidget(self.multipleCams_delay_horizontalSlider, 0, 1, 1, 1)
        self.multicam_delay_label = QtWidgets.QLabel(self.groupBox_12)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.multicam_delay_label.setFont(font)
        self.multicam_delay_label.setObjectName("multicam_delay_label")
        self.gridLayout_11.addWidget(self.multicam_delay_label, 0, 2, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_12, 1, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_10.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_10.setFont(font)
        self.groupBox_10.setObjectName("groupBox_10")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_10)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_14 = QtWidgets.QLabel(self.groupBox_10)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_9.addWidget(self.label_14, 0, 0, 1, 1)
        self.handTracking_leftHand_horizontalSlider = QtWidgets.QSlider(self.groupBox_10)
        self.handTracking_leftHand_horizontalSlider.setMinimum(1)
        self.handTracking_leftHand_horizontalSlider.setMaximum(10)
        self.handTracking_leftHand_horizontalSlider.setSingleStep(1)
        self.handTracking_leftHand_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.handTracking_leftHand_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.handTracking_leftHand_horizontalSlider.setTickInterval(0)
        self.handTracking_leftHand_horizontalSlider.setObjectName("handTracking_leftHand_horizontalSlider")
        self.gridLayout_9.addWidget(self.handTracking_leftHand_horizontalSlider, 0, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox_10)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_9.addWidget(self.label_20, 0, 2, 1, 1)
        self.horizontalLayout_3.addWidget(self.groupBox_10)
        self.groupBox_11 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_11.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_11.setFont(font)
        self.groupBox_11.setObjectName("groupBox_11")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_11)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_22 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.gridLayout_10.addWidget(self.label_22, 0, 0, 1, 1)
        self.handTracking_rightHand_horizontalSlider = QtWidgets.QSlider(self.groupBox_11)
        self.handTracking_rightHand_horizontalSlider.setMinimum(1)
        self.handTracking_rightHand_horizontalSlider.setMaximum(10)
        self.handTracking_rightHand_horizontalSlider.setSingleStep(1)
        self.handTracking_rightHand_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.handTracking_rightHand_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.handTracking_rightHand_horizontalSlider.setTickInterval(0)
        self.handTracking_rightHand_horizontalSlider.setObjectName("handTracking_rightHand_horizontalSlider")
        self.gridLayout_10.addWidget(self.handTracking_rightHand_horizontalSlider, 0, 1, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayout_10.addWidget(self.label_21, 0, 2, 1, 1)
        self.horizontalLayout_3.addWidget(self.groupBox_11)
        self.gridLayout_2.addWidget(self.groupBox_3, 1, 1, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSave_Settings = QtWidgets.QAction(MainWindow)
        self.actionSave_Settings.setEnabled(False)
        self.actionSave_Settings.setObjectName("actionSave_Settings")
        self.menuFile.addAction(self.actionSave_Settings)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Motion Tracking"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Sit To Stand"))
        self.label_6.setText(_translate("MainWindow", "Body"))
        self.label_5.setText(_translate("MainWindow", "Hip"))
        self.sitToStand_hip_label.setText(_translate("MainWindow", "-"))
        self.sitToStand_body_label.setText(_translate("MainWindow", "-"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Left Arm Reach"))
        self.label.setText(_translate("MainWindow", "Elbow"))
        self.label_2.setText(_translate("MainWindow", "Shoulder"))
        self.left_elbow_label.setText(_translate("MainWindow", "-"))
        self.left_shoulder_label.setText(_translate("MainWindow", "-"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Right Arm Reach"))
        self.label_4.setText(_translate("MainWindow", "Shoulder"))
        self.label_3.setText(_translate("MainWindow", "Elbow"))
        self.right_elbow_label.setText(_translate("MainWindow", "-"))
        self.right_shoulder_label.setText(_translate("MainWindow", "-"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Steps Tracking"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Standing Timer"))
        self.label_9.setText(_translate("MainWindow", "Hip"))
        self.label_12.setText(_translate("MainWindow", "Body"))
        self.standing_hip_label.setText(_translate("MainWindow", "-"))
        self.standing_body_label.setText(_translate("MainWindow", "-"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Side View"))
        self.label_10.setText(_translate("MainWindow", "Knee"))
        self.label_13.setText(_translate("MainWindow", "Foot"))
        self.steps_side_knee_label.setText(_translate("MainWindow", "-"))
        self.steps_side_foot_label.setText(_translate("MainWindow", "-"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Front or Rear View"))
        self.label_11.setText(_translate("MainWindow", "Sensitivity"))
        self.steps_front_rear_sensitivity_label.setText(_translate("MainWindow", "-"))
        self.groupBox_12.setTitle(_translate("MainWindow", "Multiple Cameras"))
        self.label_23.setText(_translate("MainWindow", "Delay"))
        self.multicam_delay_label.setText(_translate("MainWindow", "-"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Hand Tracking"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Left Hand"))
        self.label_14.setText(_translate("MainWindow", "Sensitivity"))
        self.label_20.setText(_translate("MainWindow", "-"))
        self.groupBox_11.setTitle(_translate("MainWindow", "Right Hand"))
        self.label_22.setText(_translate("MainWindow", "Sensitivity"))
        self.label_21.setText(_translate("MainWindow", "-"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSave_Settings.setText(_translate("MainWindow", "Save Settings"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
