import sys
from matplotlib import pyplot as plt
from traceback import format_exc

from plot_util import co_ords, delsys, wmore
from plot_util.util import Util as util

from PyQt5 import QtCore, QtWidgets, QtGui
from plot_util.gui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, QtWidgets.QWidget, Ui_MainWindow):
    """
    front-end main-window thread: handles graphical user interface

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        """ set up gui """
        self.setupUi(self)
        self.setWindowTitle("PhysiCam")

        """ create the worker thread """
        #self._main_thread = MainThread()
        #self._main_thread.start()

        self._file_name1 = str()
        self._file_name2 = str()

        self._sensor = str()
        self._side = str()

        self._max_fname_len = 70
        self._time_range = [0, 60]

        co_ords.reset()
        wmore.reset()
        delsys.reset()

        self.ChooseFilePushButton1.clicked.connect(lambda: self.choose_file(1))
        self.ChooseFilePushButton2.clicked.connect(lambda: self.choose_file(2))
        self.PlotPushButton.clicked.connect(self.plot)

        self.MotionSensorComboBox.addItem(str())
        self.MotionSensorComboBox.addItem("Wmore")
        self.MotionSensorComboBox.currentIndexChanged.connect(self.update_motion_sensor)

        self.SideComboBox.addItem(str())
        self.SideComboBox.addItem("Left")
        self.SideComboBox.addItem("Right")
        self.SideComboBox.currentIndexChanged.connect(self.update_side)

        self.FromLineEdit.editingFinished.connect(lambda: self.update_time_range(0))
        self.ToLineEdit.editingFinished.connect(lambda: self.update_time_range(1))


    def choose_file(self, id):

        if id == 1:
            self._file_name1 = QtWidgets.QFileDialog.getOpenFileName(self, "Open Motion Sensor File", "./")

            if len(self._file_name1[0]) > self._max_fname_len:
                self.FileLabel1.setText(f"...{self._file_name1[0][-self._max_fname_len:]}")
            else:
                self.FileLabel1.setText(f"{self._file_name1[0]}")

            if "left" in self._file_name1[0].lower():
                self.SideComboBox.setCurrentIndex(self.SideComboBox.findText("Left"))
            elif "right" in self._file_name1[0].lower():
                self.SideComboBox.setCurrentIndex(self.SideComboBox.findText("Right"))

            if "wmore" in self._file_name1[0].lower():
                self.MotionSensorComboBox.setCurrentIndex(self.MotionSensorComboBox.findText("Wmore"))

        if id == 2:
            self._file_name2 = QtWidgets.QFileDialog.getOpenFileName(self, "Open PhysiCam File", "./")

            if len(self._file_name2[0]) > self._max_fname_len:
                self.FileLabel2.setText(f"...{self._file_name2[0][-self._max_fname_len:]}")
            else:
                self.FileLabel2.setText(f"{self._file_name2[0]}")

            if "left" in self._file_name2[0].lower():
                self.SideComboBox.setCurrentIndex(self.SideComboBox.findText("Left"))
            elif "right" in self._file_name2[0].lower():
                self.SideComboBox.setCurrentIndex(self.SideComboBox.findText("Right"))

            if "wmore" in self._file_name2[0].lower():
                self.MotionSensorComboBox.setCurrentIndex(self.MotionSensorComboBox.findText("Wmore"))


    def update_motion_sensor(self, _):
        self._sensor = self.MotionSensorComboBox.currentText()
        print(f"{self._sensor}")


    def update_side(self, _):
        self._side = self.SideComboBox.currentText().lower()
        print(f"{self._side}")

    
    def update_time_range(self, index):
        if index == 0:
            self._time_range[0] = int(self.FromLineEdit.text())
        if index == 1:
            self._time_range[1] = int(self.ToLineEdit.text())
        print(f"time range: from {self._time_range[0]} sec to {self._time_range[1]} sec")
        

    def plot(self):

        co_ords_index = "16" if self._side == "right" else "15" if self._side == "left" else None
        print(f"hand: {self._side}, tracking index: {co_ords_index}")

        if self._sensor == "Wmore":

            try:
                start_time = co_ords.read_motion_tracking_data(
                    self._file_name2[0], self._side, co_ords_index, self._time_range
                )
                co_ords_time_p, co_ords_acc_p, co_ords_sample_rate = co_ords.process_motion_tracking_data(
                    self._time_range, lpf=True
                )

                wmore.read_wmore_sensor_data(
                    self._file_name1[0], self._time_range, co_ords_start_time=start_time
                )
                wmore_time_p, wmore_acc_p, wmore_sample_rate = wmore.process_wmore_sensor_data(
                    self._time_range, lpf=True
                )

                print(
                    "motion tracking sample rate: %.2f, wmore sample rate: %.2f"
                    % (co_ords_sample_rate, wmore_sample_rate)
                )

                plt.plot(co_ords_time_p, co_ords_acc_p)
                plt.plot(wmore_time_p, wmore_acc_p)
                plt.vlines(co_ords.co_ords_count, util.bounds[0], util.bounds[1], colors="g")
                plt.xlabel("Time (s)")
                plt.ylabel("Normalised Acceleration")
                plt.legend(["Motion Tracking", "WMORE Wearable Sensor"])
                plt.show()

            except Exception:
                print(f"{format_exc()}")

            co_ords.reset()
            wmore.reset()

        elif self._sensor == "Delsys":

            try:
                co_ords.read_motion_tracking_data(
                    self._file_name2[0], self._side, co_ords_index, self._time_range
                )
                co_ords_time_p, co_ords_acc_p, co_ords_sample_rate = co_ords.process_motion_tracking_data(
                    self._time_range, lpf=True
                )

                delsys.read_delsys_sensor_data(
                    self._file_name1[0], self._time_range
                )
                delsys_time_p, delsys_acc_p, delsys_sample_rate = delsys.process_delsys_sensor_data(
                    self._time_range, lpf=True
                )

                print(
                    "motion tracking sample rate: %.2f, delsys sample rate: %.2f"
                    % (co_ords_sample_rate, delsys_sample_rate)
                )

                """ plot data """
                plt.plot(co_ords_time_p, co_ords_acc_p)
                plt.plot(delsys_time_p, delsys_acc_p)
                plt.vlines(co_ords.co_ords_count, util.bounds[0], util.bounds[1], colors="g")
                plt.xlabel("Time (s)")
                plt.ylabel("Normalised Acceleration")
                plt.legend(["Motion Tracking", "Delsys Wearable Sensor", "Detected Count"])
                plt.show()

            except Exception as err:
                print(f"{format_exc()}")

            co_ords.reset()
            delsys.reset()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()