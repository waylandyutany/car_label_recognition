################################################################################
class Label:
    def __init__(self, label_cbbs):
        self.__cbbs = label_cbbs
        self.__text = "".join((cbb.param for cbb in label_cbbs))

    @property
    def text(self):
        return self.__text

    @property
    def cbbs(self):
        return self.__cbbs
