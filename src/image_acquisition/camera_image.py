import cv2
from image_acquisition.image_acquisition import ImageAcquisition

################################################################################
class CameraImage(ImageAcquisition):
    def __init__(self, width, height, cam_device_index):
        self.__cap = cv2.VideoCapture(cam_device_index)
        self.__cap.set(3, width)
        self.__cap.set(4, height)

    def __del__(self):
        self.__cap.release()

    @property
    def gray_image(self):
        ret, frame = self.__cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
