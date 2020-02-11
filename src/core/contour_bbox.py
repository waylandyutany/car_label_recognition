from cv2 import boundingRect

################################################################################
class CountourBBox:
    def __init__(self, param, contour):
        self.param = param
        self.__bbox = boundingRect(contour)
        (self.x, self.y, self.w, self.h) = self.__bbox
        self.__countour = contour

    @property
    def bbox(self):
        return self.__bbox

    def right_center(self, length):
        (x, y, w, h) = self.__bbox
        return (x + (w/2) + (w * length), y + (h/2))

    def is_same(self, cbb, delta):
        (delta_w, delta_h) = delta
        (_, _, _w, _h) = self.__bbox
        (_, _, w, h) = cbb.bbox
        if (abs(w - _w) <= (_w * delta_w)) and (abs(h - _h) < (_h * delta_h)):
            return True
        return False

    def is_point_in(self, point):
        (x, y, w, h) = self.__bbox
        (px, py) = point
        return (x <= px) and (px <= x+w) and (y<=py) and (py<=y+h)
