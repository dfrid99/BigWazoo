import numpy as np
import os
import cv2
from numpy import random
class Trainer:
    def __init__(self,img_size):
        self.img_size = img_size
        self.train_images = np.ndarray(shape=(0,img_size,img_size))
        self.train_labels = np.ndarray(shape=(0,1))
        self.test_images = np.ndarray(shape=(0,img_size,img_size))
        self.test_labels = np.ndarray(shape=(0,1))

    def addImages(self, path, label):
        folder, dirs, files = next(os.walk(path))
        if(files[0] == '.DS_Store'):
            files.pop(0)
        file_count = len(files)
        currLenTrain = len(self.train_images)
        currLenTest = len(self.test_images)
        train_len = int(file_count * .8)
        test_len = int(file_count * .2)
        self.train_images.resize(currLenTrain + train_len, self.img_size, self.img_size)
        self.train_labels.resize(currLenTrain + train_len,1)
        self.test_images.resize(currLenTest + test_len, self.img_size, self.img_size)
        self.test_labels.resize(currLenTest + test_len, 1)

        for i, file in enumerate(files):
            if (i < train_len):
                pos = i + currLenTrain
                self.train_images[pos] = cv2.imread(path + file, 0)
                self.train_labels[pos] = label
            else:
                pos = currLenTest + i - train_len
                self.test_images[pos] = cv2.imread(path + file, 0)
                self.test_labels[pos] = label

    def shuffleArrs(self):
        rng_state = random.get_state()
        random.shuffle(self.train_images)
        random.set_state(rng_state)
        random.shuffle(self.train_labels)

    def prepare(self):
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
        self.train_images.resize(self.train_images.shape + (1,))
        self.test_images.resize(self.test_images.shape + (1,))










