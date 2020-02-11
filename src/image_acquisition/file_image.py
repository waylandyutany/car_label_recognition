import cv2
from image_acquisition.image_acquisition import ImageAcquisition

################################################################################
class FileImage(ImageAcquisition):
    def __init__(self, file_path):
        self.__gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    @property
    def gray_image(self):
        return self.__gray_image
