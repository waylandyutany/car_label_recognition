import cv2, os, time
from core.labels_detector import LabelsDetector
from core.label import Label

################################################################################
class TimeDelta:
    def __init__(self, obj, delta_name):
        self.__start = None
        self.__obj = obj
        self.__delta_name = delta_name

    def __enter__(self):
        self.__start = time.time()

    def __exit__(self ,type, value, traceback):
        setattr(self.__obj, self.__delta_name, time.time() - self.__start)

    @property
    def delta(self):
        return self.__delta

################################################################################
class LabelsRegions:
    adaptive_tresh_method = cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    adaptive_tresh_hold_type = cv2.THRESH_BINARY
    adaptive_tresh_block_size = 11
    adaptive_tresh_C = 6
    filter_min_height = 32
    filter_min_width = 16
    min_w_h_ratio = 0.3
    max_w_h_ratio = 0.7

    def __init__(self, image_acquisition, file_path, recognitor):
        self.__src_image = image_acquisition.gray_image

        with TimeDelta(self,'time_contours_detection') as _:
            self.__all_contours, self.__tresholded_img = LabelsRegions.__detect_contours(self.__src_image)
        with TimeDelta(self,'time_contours_filtering') as _:
            self.__filtered_contours = LabelsRegions.__filter_contours(self.__all_contours)
        with TimeDelta(self,'time_labels_detection') as _:
            self.__detected_labels = self.__detect_labels(self.__filtered_contours, recognitor)

        self.time_total_detection = self.time_contours_detection\
                                    + self.time_contours_filtering\
                                    + self.time_labels_detection
################################################################################
    @classmethod
    def __detect_contours(cls, gray_img):
#        dst = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#working !!!
        tresholded_img = cv2.adaptiveThreshold(gray_img, 255, 
                                    cls.adaptive_tresh_method, 
                                    cls.adaptive_tresh_hold_type, 
                                    cls.adaptive_tresh_block_size,
                                    cls.adaptive_tresh_C)
        contours, hierarchy = cv2.findContours(tresholded_img,
                                               cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
        return contours, tresholded_img

    @classmethod
    def __filter_contours(cls, contours_to_filter):
        contours = []
        cv2_boundingRect = cv2.boundingRect
        contours_append = contours.append
        for contour in contours_to_filter:
            (x, y, w, h) = cv2_boundingRect(contour)
            if (h < cls.filter_min_height) or (w < cls.filter_min_width): continue
            w_h_ratio = w / h
            if (cls.min_w_h_ratio < w_h_ratio) and (w_h_ratio < cls.max_w_h_ratio):
                contours_append(contour)

        return contours

    def __detect_labels(self, filter_contours, recognitor):
        labels = LabelsDetector(filter_contours)
        imgs = [self.extract_image_for_recognition(cbb.bbox, recognitor.rc_dimms) for cbb in labels.yield_label_cbbs()]
        result = recognitor.detect_images(imgs)

        for i, cbb in enumerate(labels.yield_label_cbbs()):
            cbb.param = result[i]

        return [Label(label_cbbs) for label_cbbs in labels.labels_cbbs]

################################################################################
    def construct_debug_image(self, contours = None, rec_color=(0, 255, 0), con_color=(255, 0, 0)):
        cv2_boundingRect = cv2.boundingRect
        img = cv2.cvtColor(self.__tresholded_img, cv2.COLOR_GRAY2BGR)
        if contours:
            cv2.drawContours(img, contours, -1, con_color, 1)
            for contour in contours:
                (x, y, w, h) = cv2_boundingRect(contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), rec_color, thickness=2, lineType=8, shift=0)

        return img

################################################################################
    def extract_image_for_recognition(self, bbox, rc_dimms):
        (rc_width, rc_height) = rc_dimms
        (x, y, w, h) = bbox
        bw = cv2.threshold(self.__src_image[y:y+h, x:x+w], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        resized_img = cv2.resize(bw, (rc_width, rc_height), interpolation = cv2.INTER_AREA)
        return resized_img

################################################################################
    def all_contours(self):
        return self.__all_contours

    def filtered_contours(self):
        return self.__filtered_contours

    def detected_labels(self):
        return self.__detected_labels

################################################################################
    def save_labels_recognitions(self, path, name,labels, rc_dimms):
        (rc_width, rc_height) = rc_dimms
        path = os.path.join(path, name, "{}_{}".format(rc_width, rc_height))
        if not os.path.exists(path):
           os.makedirs(path)

        index = 0
        for label in labels:
            for cbb in label.cbbs:
                file_name = "{}_{}_{}_{}_{:02d}.jpg".format(cbb.param, rc_width, rc_height, name, index)
                file_path = os.path.join(path, file_name)
                cv2.imwrite(file_path, self.extract_image_for_recognition(cbb.bbox, rc_dimms))
                index += 1
                #print("Saving label recognition '{}'...".format(file_path))
