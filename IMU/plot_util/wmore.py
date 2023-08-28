import csv
import numpy as np
from scipy import interpolate as interp
from .util import Util as util

""" directories for the wmore data """
wmore_dir = "Recordings_Wmore/"

wmore_subdir = "box_and_blocks_right_0/"
wmore_data = "box_and_blocks_right_0"


wmore_time_range = (0, 60)


def read_wmore_sensor_data(data, co_ords_start_time=None):
    with open(data) as wmore_file:
        wmore_reader = csv.DictReader(wmore_file)

        count = 0
        prev_time = None
        for i, row in enumerate(wmore_reader):
            valid = int(row["valid"])
            year = int(row["g_year"])

            if valid and year == 23:
                hund = int(row["g_hund"])
                sec = int(row["g_second"])
                minute = int(row["g_minute"])
                hr = int(row["g_hour"])

                if count == 0:
                    pass
                elif count == 1:
                    start_time = hund*0.01 + sec + minute*60 + hr*3600
                    if start_time < co_ords_start_time:
                        continue

                else:
                    if count == 2:
                        print(
                            f"wmore start time: {start_time}, co-ords start time {co_ords_start_time}"
                        )

                    curr_time = hund * 0.01 + sec + minute * 60 + hr * 3600
                    time = curr_time - start_time

                    if (
                        time < wmore_time_range[1]
                        and time > wmore_time_range[0]
                        and time != prev_time
                    ):
                        wmore_time.append(time)
                        wmore_acc_x.append(int(row["ax"]))
                        wmore_acc_y.append(int(row["ay"]))
                        wmore_acc_z.append(int(row["az"]))

                        prev_time = time

                count += 1


def process_wmore_sensor_data(lpf=False):

    wmore_sample_rate = len(wmore_time) / (wmore_time_range[1] - wmore_time_range[0])
    f_cutoff, f_order = 3, 3

    if lpf:
        wmore_acc_x_filt = util.low_pass_filter(wmore_acc_x, f_cutoff, wmore_sample_rate, f_order)
        wmore_acc_y_filt = util.low_pass_filter(wmore_acc_y, f_cutoff, wmore_sample_rate, f_order)
        wmore_acc_z_filt = util.low_pass_filter(wmore_acc_z, f_cutoff, wmore_sample_rate, f_order)
    else:
        wmore_acc_x_filt = wmore_acc_x
        wmore_acc_y_filt = wmore_acc_y
        wmore_acc_z_filt = wmore_acc_z

    wmore_acc = util.remove_gravity(util.get_magnitude_3d(wmore_acc_x_filt, wmore_acc_y_filt, wmore_acc_z_filt), 2**14)

    new_wmore_time = list()
    for t in wmore_time:
        new_wmore_time.append(t + 1)
        #new_wmore_time.append(t)

    if lpf:
        new_wmore_acc = util.low_pass_filter(wmore_acc, 1, wmore_sample_rate, 1)
    else:
        new_wmore_acc = wmore_acc

    new_wmore_acc = util.normalise(new_wmore_acc, util.bounds)

    """ smooth sensor data """
    wmore_spl = interp.make_interp_spline(np.array(new_wmore_time), np.array(new_wmore_acc))
    np_wmore_time = np.linspace(min(new_wmore_time), max(new_wmore_time), 1000)
    np_wmore_acc = wmore_spl(np_wmore_time)

    return np_wmore_time, np_wmore_acc, wmore_sample_rate


def reset():
    global wmore_time, wmore_acc
    global wmore_acc_x, wmore_acc_y, wmore_acc_z

    wmore_time = list()
    wmore_acc = list()

    wmore_acc_x = list()
    wmore_acc_y = list()
    wmore_acc_z = list()