import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
import h5py
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, folder="data/jpg"):
        """Dataset initialization
        Args:
            folder (str, optional): Path of the folder where images are. Images
                must be in jpg format
        """
        self.path = folder
        self.all_images = [f for f in sorted(listdir(
            self.path)) if isfile(join(self.path, f))]

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
        if type(idx) is str:
            return self.get_image_by_name(idx)
        else:
            return self.all_images[idx]

    def read_image(self, image_path, scale=1.):
        """Reads an image from the image folder

        Args:
            image_path (TYPE): Image path
            scale (float, optional): Scale factor for image resizing. Default is 1, which means no scaling.

        Returns:
            Image: as an np.array

        Raises:
            FileNotFoundError: If the image is not found or can't be read
        """
        if not (isfile(image_path)):
            image_path = os.path.abspath(join(self.path, image_path))

        if not (isfile(image_path)):
            raise FileNotFoundError(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_image_by_name(self, image_name=None):
        """Returns an image based on its file name

        Args:
            image_name (None, optional): Imafe file name (without extension)

        Returns:
            Image: as an np.array
        """
        return self.read_image(image_name)

    def get_random_image(self):
        """Returns a random image from the dataset

        Returns:
            Image: as an np.array
        """
        return self.get_image_by_name(random.choice(self.all_images))

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
        """Extracts the features (descriptor) from an image. The features
        are also stored on the disk.

        Args:
            image_path (TYPE): This can be the image path or image name

        Returns:
            features: List of descriptors for the image
        """
        # check if the feature can be retrieved from disk
        if self.is_stored(image_path):
            hdf5_path = os.path.join(
                self.path, "..", "features_%s.hdf5" % self.sift_implementation)
            with h5py.File(hdf5_path, "r") as file:
                image_id = self.get_image_id(image_path)
                features = np.array(file[image_id])
            return features  # / np.linalg.norm(features)

        # if not, calculate the feature
        image = self.read_image(image_path)
        features = self.descriptor.describe(image)

        # once, calculated, store the features if they're not on disk
        # you can force restoring with force=True
        self.store_features(image_path, features)

        # return
        return features

    def store_features(self, image_path, features, force=False):
        """Stores the extracted features on the disk for subsequent retrieval

        Args:
            image_path (str): Image name or path
            features (list): List of descriptors
            force (bool, optional): If `True`, overwrites previously stored features

        Returns:
            None
        """
        if self.is_stored(image_path) and not force:
            return
        image_id = self.get_image_id(image_path)
        with h5py.File(os.path.join(self.path, "..", "features_%s.hdf5" % self.sift_implementation), "a") as file:
            features = np.array(features)
            file.create_dataset(image_id, features.shape, data=features)
        return

    def is_stored(self, image_path):
        """Helper function to check wether the descriptors for a given image
        have been already computed and stores

        Args:
            image_path (str): Image name or path

        Returns: - `False` if the image is not present in the features database
                 - `list` of features if the image descriptors are present in the database
        """
        hdf5_path = os.path.join(
            self.path, "..", "features_%s.hdf5" % self.sift_implementation)
        if not os.path.isfile(hdf5_path):
            return False
        with h5py.File(hdf5_path, "r") as file:
            return self.get_image_id(image_path) in file
        return False

    @staticmethod
    def show_image(img, gray=False, **kwargs):
        """Displays an image

        Args:
            img (np.array): The image to show
            gray (bool, optional): Wether to use grayscale colormap
            **kwargs: Extra options to the plt.imshow() function
        """
        if not gray:
            plt.imshow(img, aspect="equal", **kwargs)
        else:
            plt.imshow(img, aspect="equal", cmap="gray", **kwargs)
