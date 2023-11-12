"""
Created on Tue Nov  7 22:35:49 2023

@author: Bren Guzmán

"""

import struct
import numpy as np
from array import array

class Dataframe(object):
    def __init__(self, images_filepath, labels_filepath):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath

    def load_data(self):
        labels = []
        with open(self.labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))
        
        with open(self.images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.array(array("B", file.read()))
        
        images = image_data.reshape(size, rows * cols)
        
        return images, labels
