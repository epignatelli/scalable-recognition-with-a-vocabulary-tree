import os
import h5py
from .. import utils
import numpy as np


class DescriptorBase(object):
    def __init__(self, store_root=None):
        if store_root is not None:
            self._storage = os.path.join(store_root, "features_%s.hdf5" % self.__class__.__name__)
        else:
            self._storage = None

    def __call__(self, image):
        image_id = utils.get_image_id(image)

        if not self.is_stored(image_id) and self._storage is not None:
            features = self.describe(image)
            self.store(image_id, features)

        return self.load(image_id)

    def load(self, image_id):
        with h5py.File(self._storage, "r") as file:
                return np.array(file[image_id])

    def store(self, image_id, features, force=False, store_root="data"):
        """Stores the extracted features on the disk for subsequent retrieval

        Args:
            image_path (str): Image name or path
            features (list): List of descriptors
            force (bool, optional): If `True`, overwrites previously stored features

        Returns:
            None
        """
        if self._storage is None:
            return

        if self.is_stored(image_id) and not force:
            return
        with h5py.File(self._storage, "a") as file:
            features = np.array(features)
            file.create_dataset(image_id, features.shape, data=features)
        return

    def is_stored(self, image_id, store_root="data"):
        """Helper function to check wether the descriptors for a given image
        have been already computed and stores

        Args:
            image_path (str): Image name or path

        Returns: - `False` if the image is not present in the features database
                 - `list` of features if the image descriptors are present in the database
        """
        if not os.path.isfile(self._storage):
            return False
        with h5py.File(self._storage, "r") as file:
            return image_id in file
        return False
