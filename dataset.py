import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
import h5py
import matplotlib.pyplot as plt
from features import Descriptor
from ezsift import EzSIFT


class Dataset():
    def __init__(self, folder="data/jpg", sift_implementation="pytorch"):
        self.path = folder
        self.all_images = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        self.sift_implementation = sift_implementation
        if sift_implementation.lower() == "ezsift":
            self.descriptor = EzSIFT()
        else:
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

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        return self.all_images[idx]

    def read_image(self, image_path, gray=False):
        if not (isfile(image_path)):
            image_path = os.path.abspath(join(self.path, image_path))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        if gray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.float32(gray)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_image_by_name(self, image_name=None, gray=True):
        path = os.path.join(self.path, image_name)
        print(path)
        return self.read_image(path)

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

    def extract_features(self, image_path):
        # check if the feature can be retrieved from disk
        if self.is_stored(image_path):
            hdf5_path = os.path.join(self.path, "..", "features_%s.hdf5" % self.sift_implementation)
            with h5py.File(hdf5_path, "r") as file:
                image_id = self.get_image_id(image_path)
                features = np.array(file[image_id])
            return features / np.linalg.norm(features)

        # if not, calculate the feature
        image = self.read_image(image_path, gray=False)
        features = self.descriptor.describe(image)

        # once, calculated, store the features if they're not on disk
        # you can force restoring with force=True
        self.store_features(image_path, features)

        # return
        return features

    def store_features(self, image_path, features, force=False):
        if self.is_stored(image_path) and not force:
            return
        image_id = self.get_image_id(image_path)
        with h5py.File(os.path.join(self.path, "..", "features_%s.hdf5" % self.sift_implementation), "a") as file:
            features = np.array(features)
            file.create_dataset(image_id, features.shape, data=features)
        return

    def is_stored(self, image_path):
        hdf5_path = os.path.join(self.path, "..", "features_%s.hdf5" % self.sift_implementation)
        if not os.path.isfile(hdf5_path):
            return False
        with h5py.File(hdf5_path, "r") as file:
            return self.get_image_id(image_path) in file
        return False

    @staticmethod
    def show_image(img, gray=False, **kwargs):
        if not gray:
            plt.imshow(img, aspect="equal", **kwargs)
        else:
            plt.imshow(img, aspect="equal", cmap="gray", **kwargs)
