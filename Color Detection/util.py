import numpy as np
import cv2

# Function to get HSV color limits for a given BGR color to pass to main function for color detection
def get_limits(color):

    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimmit = hsvC[0][0][0] - 10, 100, 100
    upperLimmit = hsvC[0][0][0] + 10, 255, 255

    lowerLimmit = np.array(lowerLimmit, dtype=np.uint8)
    upperLimmit = np.array(upperLimmit, dtype=np.uint8)

    return lowerLimmit, upperLimmit