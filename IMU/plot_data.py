import sys
from matplotlib import pyplot as plt
from plot_util import co_ords, delsys, wmore
from plot_util.util import Util as util

""" program parameters """
usage = 'usage: plot_data.py [sensor: "delsys" | "wmore"] [handedness: "right" | "left"]'

""" exit if no parameters provided """
try:
    sensor = sys.argv[1]
    handedness = sys.argv[2]
except:
    print(usage)
    sys.exit()

""" exit if index is out of range """
if (sensor != "delsys" and sensor != "wmore") or (handedness != "right" and handedness != "left"):
    print(usage)
    sys.exit()

""" 
set left or right hand 

"""
co_ords_index = "16" if handedness == "right" else "15" if handedness == "left" else None
print(f"hand: {handedness}, tracking index: {co_ords_index}")


def main():
    if sensor == "wmore":
        data = f"{wmore.wmore_dir}{wmore.wmore_data}"

        start_time = co_ords.read_motion_tracking_data(data, handedness, co_ords_index, wmore.wmore_time_range)
        co_ords_time_p, co_ords_acc_p, co_ords_sample_rate = co_ords.process_motion_tracking_data(
            wmore.wmore_time_range, lpf=True
        )

        wmore.read_wmore_sensor_data(co_ords_start_time=start_time)
        wmore_time_p, wmore_acc_p, wmore_sample_rate = wmore.process_wmore_sensor_data(lpf=True)

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

    elif sensor == "delsys":
        data = f"{delsys.delsys_dir}{delsys.delsys_data}"

        co_ords.read_motion_tracking_data(data, handedness, co_ords_index, delsys.delsys_time_range)
        co_ords_time_p, co_ords_acc_p, co_ords_sample_rate = co_ords.process_motion_tracking_data(
            delsys.delsys_time_range, lpf=True
        )

        delsys.read_delsys_sensor_data(data)
        delsys_time_p, delsys_acc_p, delsys_sample_rate = delsys.process_delsys_sensor_data(lpf=True)

        print(
            f"motion tracking sample rate: {co_ords_sample_rate}, delsys sample rate: {delsys_sample_rate}"
        )

        """ plot data """
        plt.plot(co_ords_time_p, co_ords_acc_p)
        plt.plot(delsys_time_p, delsys_acc_p)
        plt.vlines(co_ords.co_ords_count, util.bounds[0], util.bounds[1], colors="g")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalised Acceleration")
        plt.legend(["Motion Tracking", "Delsys Wearable Sensor", "Detected Count"])
        plt.show()


if __name__ == "__main__":
    main()
