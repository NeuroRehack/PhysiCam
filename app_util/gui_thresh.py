# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/gui_thresh.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 786, 550))
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
        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 1, 1, 1, 1)
        self.sitToStand_hipAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_4)
        self.sitToStand_hipAngle_horizontalSlider.setMinimum(90)
        self.sitToStand_hipAngle_horizontalSlider.setMaximum(180)
        self.sitToStand_hipAngle_horizontalSlider.setSingleStep(1)
        self.sitToStand_hipAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sitToStand_hipAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sitToStand_hipAngle_horizontalSlider.setTickInterval(0)
        self.sitToStand_hipAngle_horizontalSlider.setObjectName("sitToStand_hipAngle_horizontalSlider")
        self.gridLayout_3.addWidget(self.sitToStand_hipAngle_horizontalSlider, 1, 2, 1, 1)
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
        self.leftArmReach_elbowAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_5)
        self.leftArmReach_elbowAngle_horizontalSlider.setMinimum(90)
        self.leftArmReach_elbowAngle_horizontalSlider.setMaximum(180)
        self.leftArmReach_elbowAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.leftArmReach_elbowAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.leftArmReach_elbowAngle_horizontalSlider.setObjectName("leftArmReach_elbowAngle_horizontalSlider")
        self.gridLayout_4.addWidget(self.leftArmReach_elbowAngle_horizontalSlider, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 4, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 0, 0, 1, 1)
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
        self.rightArmReach_shoulderAngle_horizontalSlider = QtWidgets.QSlider(self.groupBox_6)
        self.rightArmReach_shoulderAngle_horizontalSlider.setMinimum(0)
        self.rightArmReach_shoulderAngle_horizontalSlider.setMaximum(90)
        self.rightArmReach_shoulderAngle_horizontalSlider.setProperty("value", 0)
        self.rightArmReach_shoulderAngle_horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rightArmReach_shoulderAngle_horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rightArmReach_shoulderAngle_horizontalSlider.setObjectName("rightArmReach_shoulderAngle_horizontalSlider")
        self.gridLayout_5.addWidget(self.rightArmReach_shoulderAngle_horizontalSlider, 1, 1, 1, 1)
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
        self.horizontalLayout_2.addWidget(self.groupBox_7)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_2.addWidget(self.groupBox_8)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_9.setFont(font)
        self.groupBox_9.setObjectName("groupBox_9")
        self.horizontalLayout_2.addWidget(self.groupBox_9)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_10.setFont(font)
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_3.addWidget(self.groupBox_10)
        self.groupBox_11 = QtWidgets.QGroupBox(self.groupBox_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.groupBox_11.setFont(font)
        self.groupBox_11.setObjectName("groupBox_11")
        self.horizontalLayout_3.addWidget(self.groupBox_11)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Motion Tracking"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Sit To Stand"))
        self.label_5.setText(_translate("MainWindow", "Hip Angle"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Left Arm Reach"))
        self.label_2.setText(_translate("MainWindow", "Shoulder"))
        self.label.setText(_translate("MainWindow", "Elbow"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Right Arm Reach"))
        self.label_3.setText(_translate("MainWindow", "Elbow"))
        self.label_4.setText(_translate("MainWindow", "Shoulder"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Steps Tracking"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Standing Timer"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Left Steps"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Right Steps"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Hand Tracking"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Left Hand"))
        self.groupBox_11.setTitle(_translate("MainWindow", "Right Hand"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())