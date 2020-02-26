import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt
from .features import Descriptor


class Dataset():
    def __init__(self, folder="data/jpg"):
        self.path = folder
        self.all_images = [f for f in listdir(
            self.path) if isfile(join(self.path, f))]
        self.descriptor = Descriptor()

    def __str__(self):
        images = []
        for i in range(len(self.all_images)):
            images.append(self.all_images[i])
            if i == 5:
                break
        return str(images)

    def __repr__(self):
        return str(self)

    def get_image_by_name(self, image_name=None, gray=True):
        print(self.path + '/' + image_name)
        image = cv2.imread(self.path + '/' + image_name)
        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        if gray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.float32(gray)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_random_image(self, gray=False):
        return self.get_image_by_name(random.choice(self.all_images), gray)

    def get_image_id(self, image_path):
        """
        Given an image path, returns the id of the image: the numerical part of the filename
        Args:
            image_path (str): path of the image
        Returns:
            (str): the id of the image as string
        """
        return os.path.splitext(os.path.basename(image_path))[0]

    def extract_features(self, image):
        keypoints, blobs = self.descriptor.find_keypoints(image)
        patches = self.descriptor.extract_MSER_patches(image, blobs)
        descriptors = [self.descriptor.describe(patch) for patch in patches]
        return descriptors

    def show_image(img, gray=False):
        if not gray:
            plt.imshow(img, aspect="equal")
        else:
            plt.imshow(img, aspect="equal", cmap="gray")
