from core.contour_bbox import CountourBBox

################################################################################
class LabelsDetector:
    min_letters_in_label = 5
    same_box_treshold_percentage = (0.33, 0.15)

    @classmethod
    #container must contains cbbs sorted from left to right based on x !!!
    def find_next_cbb(cls, container,  from_index):
        from_cbb = container[from_index]
        for i in range(from_index + 1, len(container)):
            cbb = container[i]
            #do early rejections
            if (cbb.param < 0): continue
            if (cbb.x > from_cbb.x + (from_cbb.w * 3)): break

            #check if right center point(near or far) is in next box
            if cbb.is_same(from_cbb,cls.same_box_treshold_percentage)\
                and (cbb.is_point_in(from_cbb.right_center(1)) or cbb.is_point_in(from_cbb.right_center(2))):
                return i

        return -1

    def __init__(self, contours):
        self.__countour_bboxes = []
        self.__labels_cbbs = []

        for i, contour in enumerate(contours):
            cbb = CountourBBox(i, contour)
            self.__countour_bboxes.append(cbb)

        self.__countour_bboxes.sort(key=lambda cbb: cbb.x)

        for i in range(0, len(self.__countour_bboxes)):
            if self.__countour_bboxes[i].param < 0: continue
            label_cbbs = [self.__countour_bboxes[i]]
            
            next_cbb_index = LabelsDetector.find_next_cbb(self.__countour_bboxes, i)
            while next_cbb_index >= 0:
                label_cbbs.append(self.__countour_bboxes[next_cbb_index])
                next_cbb_index = LabelsDetector.find_next_cbb(self.__countour_bboxes, next_cbb_index)

            if len(label_cbbs) >= LabelsDetector.min_letters_in_label:
                for cbb in label_cbbs:
                    cbb.param = -1
                self.__labels_cbbs.append(label_cbbs)

    @property
    def labels_cbbs(self):
        return self.__labels_cbbs

################################################################################
    def yield_bboxes(self):
        return self.__yield_label_bboxes()

    def __yield_all_bboxes(self):
        for cbb in self.__countour_bboxes:
            yield cbb.bbox

    def __yield_label_bboxes(self):
        for label in self.__labels_cbbs:
            for cbb in label:
                yield cbb.bbox

    def yield_label_cbbs(self):
        for label in self.__labels_cbbs:
            for cbb in label:
                yield cbb

################################################################################
