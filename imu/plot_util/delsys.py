import csv
import numpy as np
from scipy import interpolate as interp
from .util import Util as util

delsys_dir = "Recordings_Delsys/"

delsys_data = "230710_14-53-47"

#delsys_time_range = (10, 30)


def read_delsys_sensor_data(data, time_range):
    """
    read wearable sensor data

    """
    with open(data) as delsys_file:
        delsys_reader = csv.DictReader(delsys_file)
        for row in delsys_reader:
            time = row["X[s]"]
            if time == "":
                break

            time = float(time)
            if time < time_range[1] and time > time_range[0]:
                delsys_time.append(float(time))
                delsys_acc_x.append(float(row["Avanti sensor 1: ACC.X 1"]))
                delsys_acc_y.append(float(row["Avanti sensor 1: ACC.Y 1"]))
                delsys_acc_z.append(float(row["Avanti sensor 1: ACC.Z 1"]))


def process_delsys_sensor_data(time_range, lpf=False):
    """get magnitude of acceleration for delsys sensor data"""
    delsys_acc = util.remove_gravity(util.get_magnitude_3d(delsys_acc_x, delsys_acc_y, delsys_acc_z), 1)
    delsys_sample_rate = len(delsys_time) / (time_range[1] - time_range[0])

    new_delsys_time = delsys_time

    if lpf:
        new_delsys_acc = util.low_pass_filter(delsys_acc, 1, delsys_sample_rate, util.filt_order)
    else:
        new_delsys_acc = delsys_acc

    new_delsys_acc = util.normalise(new_delsys_acc, util.bounds)

    """ smooth wearable sensor data """
    delsys_spl = interp.make_interp_spline(np.array(new_delsys_time), np.array(new_delsys_acc))
    np_delsys_time = np.linspace(min(new_delsys_time), max(new_delsys_time), 1000)
    np_delsys_acc = delsys_spl(np_delsys_time)

    return np_delsys_time, np_delsys_acc, delsys_sample_rate


def reset():
    global delsys_time, delsys_acc
    global delsys_acc_x, delsys_acc_y, delsys_acc_z

    delsys_time = list()
    delsys_acc = list()

    delsys_acc_x = list()
    delsys_acc_y = list()
    delsys_acc_z = list()