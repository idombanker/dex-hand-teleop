import sys
import time

import cv2
import numpy as np

from threading import Thread
from hand_detector.hand_monitor import Record3DSingleHandMotionControl


class RealTimeHandControl:
    def __init__(self, hand_mode: str):
        self.motion_control = Record3DSingleHandMotionControl(hand_mode)


    def run(self):
        # global runtimes

        while True:
            # start_time = time.time()  # Record the start time
    
            success, output = self.motion_control.normal_step()
            if success:
                joints = output["joint"]


            else:
                print("No hand detected.")


if __name__ == "__main__":
    hand_mode = "right_hand"  # Change to "left_hand" if needed
    visualizer = RealTimeHandControl(hand_mode)
    visualizer.run()
