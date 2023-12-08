import cv2 as cv
import numpy as np

""" aruco marker sizes """
ARUCO_SIZES = [4, 5, 6, 7]

mtx = np.array(
    [
        [500.0, 0.0,  300.0],
        [0.0, 500.0, 300.0],
        [0.0, 0.0, 1.0],
    ]
)
dist = np.array(
    [
        [0.01, 0.01, 0.01, 0.01, 0.01]
    ]
)

class Aruco:
    def __init__(self):
        pass

    def find_aruco(self, img, detected):
        """
        looks for aruco markers of various sizes

        """
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for size in ARUCO_SIZES:
            if size == ARUCO_SIZES[0]:
                aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
            elif size == ARUCO_SIZES[1]:
                aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
            elif size == ARUCO_SIZES[2]:
                aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_1000)
            else:
                aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_7X7_1000)

            aruco_params = cv.aruco.DetectorParameters()
            detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
            self._corners, self._ids, self._rejected = detector.detectMarkers(img)

            # TO-DO: draw aruco orientation
            if len(self._corners) > 0:
                for i in range(0, len(self._ids)):
                
                    rvec, tvec, mp = cv.aruco.estimatePoseSingleMarkers(self._corners[i], 0.03, mtx, dist)
                    print(f"Pose Estimation - Marker {i + 1}:\nRotation Vector (rvec): {rvec}\nTranslation Vector (tvec): {tvec}")
                    img = cv.drawFrameAxes(img, mtx, dist, rvec, tvec, 1, 3)

            img = self.display_aruco(img, detected)

        return img

    def display_aruco(self, img, detected):
        """
        draws the aruco markers in the frame
        called from the find_aruco() function

        """
        self._cropped = False

        if len(self._corners) > 0:
            self._ids = self._ids.flatten()

            """ iterate though all detected aruco markers """
            for i, (marker_corner, marker_id) in enumerate(zip(self._corners, self._ids)):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                """ converts values to integers """
                top_right = (int(top_right[0]), int(top_right[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                """ draws box around aruco marker """
                #cv.line(img, top_left, top_right, util.GREEN, 2)
                #cv.line(img, top_right, bottom_right, util.GREEN, 2)
                #cv.line(img, bottom_right, bottom_left, util.GREEN, 2)
                #cv.line(img, bottom_left, top_left, util.GREEN, 2)

                """ find the centres of the aruco markers """
                cX = int((top_left[0] + bottom_right[0]) / 2.0)
                cY = int((top_left[1] + bottom_right[1]) / 2.0)
                #cv.circle(img, (cX, cY), 2, util.RED, 5, -1)

                """ id the detected markers """
                marker_pos = (top_left[0], top_left[1] - 10)
                #cv2.putText(img, str(marker_id), marker_pos, 0, 0.5, BLUE, 1)
                detected.append((marker_id, cX, cY))

        return img