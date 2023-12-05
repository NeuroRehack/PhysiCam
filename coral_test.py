"""
script to test the functionality of the coral tpu library

"""

__author__ = "Mike Smith"
__email__ = "dongming.shi@uqconnect.edu.au"
__date__ = "05/12/2023"


import time
import cv2 as cv

from coral.coral import Coral


def display_frame_rate(img, frame_times):
    frame_times["curr time"] = time.time()
    frame_rate = 1 / (frame_times["curr time"] - frame_times["prev time"])
    frame_times["prev time"] = frame_times["curr time"]

    img = cv.flip(img, 1)
    cv.putText(img, str(int(frame_rate)), (10, 30), 0, 1, (255, 255, 0))
    img = cv.flip(img, 1)
    return img


def main():
    
    """ open video or webcam """
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    """ frame rate (for debugging) """
    frame_times = {"curr time": 0, "prev time": 0}

    # create coral edge tpu instance
    coral = Coral()

    """ main while loop """
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            print("can't connect to camera :(")
            break

        landmarks = coral.get_landmarks(img)
        img = coral.display_landmarks(img, landmarks)

        """ display frame rate """
        img = display_frame_rate(img, frame_times)
        
        """ flip frame horizontally """
        img = cv.flip(img, 1)

        """ show image """
        cv.imshow("Image", img)

        """ press "q" to exit program """
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
