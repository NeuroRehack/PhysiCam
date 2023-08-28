import csv
import numpy as np
from scipy import interpolate as interp
from .util import Util as util

co_ords_dir = "Recordings_PhysiCam/"


def read_motion_tracking_data(data, side, index, time_range):
    """
    read motion tracking co-ordinate data

    """
    prev_count = 0
    start_time = None

    with open(data) as co_ords_file:
        co_ords_reader = csv.DictReader(co_ords_file)
        for i, row in enumerate(co_ords_reader):
            hour, minute, second = row["time"].split(":")
            time = int(hour)*3600 + int(minute)*60 + float(second)

            try:
                if i == 0:
                    sys_time = row["system time"].split("_")[1].split(":")
                    start_time = int(sys_time[0])*3600 + int(sys_time[1])*60 + float(sys_time[2])
            except KeyError as err:
                print(err)

            if time < time_range[1] and time > time_range[0]:
                curr_count = row[f"{side} hand"]
                if curr_count != prev_count:
                    co_ords_count.append(time)
                    prev_count = curr_count

                """ ignore points with low visibility score """
                vis_thresh = 80
                try:
                    vis = int(row[index].strip("()").split(",")[3])
                except:
                    vis = 0
                ignore = True if vis < vis_thresh else False

                if not ignore:
                    try:
                        co_ords_time.append(time)
                        co_ords_pos_x.append(int(row[index].strip("()").split(",")[0]))
                        co_ords_pos_y.append(int(row[index].strip("()").split(",")[1]))
                        co_ords_pos_z.append(int(row[index].strip("()").split(",")[2]))
                        ignore = False
                    except:
                        ignore = True

                '''if ignore:
                    co_ords_pos_x.append(None)
                    co_ords_pos_y.append(None)
                    co_ords_pos_z.append(None)'''

    return start_time


def process_motion_tracking_data(time_range, lpf=False):
    co_ords_sample_rate = len(co_ords_time) / (time_range[1] - time_range[0])

    if lpf:
        co_ords_pos_x_filt = util.low_pass_filter(co_ords_pos_x, 3, co_ords_sample_rate, util.filt_order)
        co_ords_pos_y_filt = util.low_pass_filter(co_ords_pos_y, 3, co_ords_sample_rate, util.filt_order)
    else:
        co_ords_pos_x_filt = co_ords_pos_x
        co_ords_pos_y_filt = co_ords_pos_y

    """ derive positions to get velocity """
    co_ords_vel_x = util.derive(co_ords_time, co_ords_pos_x_filt)
    co_ords_vel_y = util.derive(co_ords_time, co_ords_pos_y_filt)

    """ derive velocity to get acceleration """
    co_ords_acc_x = util.derive(co_ords_time, co_ords_vel_x)
    co_ords_acc_y = util.derive(co_ords_time, co_ords_vel_y)

    """ get magnitude of acceleration for motion tracking co-ords """
    co_ords_acc = util.get_magnitude_2d(co_ords_acc_x, co_ords_acc_y)

    """ remove all invalid data-points eg: None """
    int_co_ords_time, int_co_ords_acc = list(), list()
    util.remove_none(co_ords_time, co_ords_acc, int_co_ords_time, int_co_ords_acc)

    new_co_ords_time = int_co_ords_time

    if lpf:
        new_co_ords_acc = util.low_pass_filter(int_co_ords_acc, 1, co_ords_sample_rate, util.filt_order)
    else:
        new_co_ords_acc = int_co_ords_acc


    new_co_ords_acc = util.normalise(new_co_ords_acc, util.bounds)

    """ smooth motion tracking data """
    co_ords_spl = interp.make_interp_spline(np.array(new_co_ords_time), np.array(new_co_ords_acc))
    np_co_ords_time = np.linspace(min(new_co_ords_time), max(new_co_ords_time), 1000)
    np_co_ords_acc = co_ords_spl(np_co_ords_time)

    return np_co_ords_time, np_co_ords_acc, co_ords_sample_rate

def reset():
    global co_ords_time, co_ords_acc, co_ords_count
    global co_ords_pos_x, co_ords_pos_y, co_ords_pos_z
    global co_ords_vel_x, co_ords_vel_y, co_ords_vel_z
    global co_ords_acc_x, co_ords_acc_y, co_ords_acc_z

    co_ords_time = list()
    co_ords_acc = list()
    co_ords_count = list()

    co_ords_pos_x = list()
    co_ords_pos_y = list()
    co_ords_pos_z = list()

    co_ords_vel_x = list()
    co_ords_vel_y = list()
    co_ords_vel_z = list()

    co_ords_acc_x = list()
    co_ords_acc_y = list()
    co_ords_acc_z = list()
