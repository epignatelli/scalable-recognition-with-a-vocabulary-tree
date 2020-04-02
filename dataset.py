import cv2
import os
from os import listdir
from os.path import isfile, join
import random
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
