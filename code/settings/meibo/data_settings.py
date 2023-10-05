import os
import numpy as np


class DataSettings(object):

    def __init__(self):

        self.BASE_PATH = os.path.abspath(
            os.path.join(
                __file__,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir))
        # self.CLASS_WEIGHTS = np.loadtxt(os.path.join(self.BASE_PATH, 'data',
        #                                              'metadata',
        #                                              'class_weights.txt'),
        #                                dtype='float', delimiter=',')[:, 1]
        self.CLASS_WEIGHTS = None
        # Assign it to None in order to disable class weighting

        self.MAX_N_OBJECTS = 25

        self.N_CLASSES = 1 + 2
