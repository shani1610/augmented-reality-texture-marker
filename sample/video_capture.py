import cv2
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from objloader_simple import *
from collections import deque

class VideoCapture:
    """bufferless VideoCapture
    """

    def __init__(self, name, res=(320, 240)):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, res[0])
        self.cap.set(4, res[1])
        self.q = deque()
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "Failed to open capture"

    def _reader(self):
        """read frames as soon as they are available, keeping only most recent one
        """

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[error] ret")
                break

            self.q.append(frame)

            self.status = "capture"

            while len(self.q) > 1:
                self.q.popleft()

        self.status = "failed"

    def read(self):
        return self.q[-1]

    def release(self):
        self.cap.release()