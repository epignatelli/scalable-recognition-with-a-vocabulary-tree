import numpy as np
import os
import cv2
import h5py
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .. import utils
from .descriptor_base import DescriptorBase


class Orb(DescriptorBase):
    def __init__(self, patch_size=65):
        super(Orb, self).__init__("data")
        self.patch_size = (int(patch_size), int(patch_size))
        self.orb = cv2.ORB.create(1500, nlevels=32)

    def describe(self, image):
        """
        Computes the ORB descriptor on the given path.

        Args:
            image (str): Image name

        Returns:
            list: List of descriptors for the image. If no keypoints are found,
                  the list will contain a single descriptor full of zeros
        """
        # find the keypoints and descriptors with ORB (like SIFT)
        kp, desc = self.orb.detectAndCompute(image, None)
        desc = np.array(desc, dtype=np.float32)
        if desc.size <= 1:
            desc = np.zeros((1, 32))
        return desc

    def extract_patches(self, img, keypoints):
        """Extracts the patches associated to the keypoints of a
        descriptors.

        Args:
            img (np.array): Image from which the patches are extracted
            keypoints (cv2.KeyPoint): Keypoints relative to the regions to extract
        Returns:
            (np.array) -- List of patches
        """
        patches = []
        height, width, _ = img.shape
        for kp in keypoints:
            mask = np.zeros((height, width), np.uint8)

            pt = (int(kp.pt[0]), int(kp.pt[1]))

            cv2.circle(mask, pt, int(kp.size), (255, 255, 255), thickness=-1)

            # Apply Threshold
            _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Find Contour
            contours = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x, y, w, h = cv2.boundingRect(contours[0][0])

            # Crop masked_data
            feat_patch = img[y:y+h, x:x+w]

            # Adding to the features
            patches.append(feat_patch)
        return patches

    @staticmethod
    def show_random_descriptors(img, keypoints, patches, descriptors, N=5):
        """
        Shows N descriptors with the corresponding patches, taken at random.

        Args:
            img (np.array): Main image
            patches (list): List of image patches
            descriptors (list): List of descriptors
            N (int, optional): Number of patches to show, default to 5
        """
        # Getting random keypoints
        random_idx = [random.randint(0, len(patches) - 1) for n in range(N)]
        some_patches = [patches[i] for i in random_idx]
        some_descriptors = [descriptors[i] for i in random_idx]

        # Setting up axes
        fig = plt.figure(constrained_layout=True, figsize=(15, 8))
        gs = GridSpec(N, 8, figure=fig)

        ax1 = fig.add_subplot(gs[:, :-2])
        img2 = cv2.drawKeypoints(img, keypoints, None,
                                 color=(0, 255, 0), flags=4)
        ax1.set_title("Image")
        plt.imshow(img2)
        plt.axis('off')

        # Showing the patched with their desctiptors
        for n in range(N):
            # Getting and showing patch
            patch = some_patches[n]
            ax = fig.add_subplot(gs[n, -2])
            if n == 0:
                ax.set_title("Patch")
            plt.imshow(patch)
            plt.axis('off')

            # Getting descriptor and plotting it
            ax = fig.add_subplot(gs[n, -1])
            if n == 0:
                ax.set_title("Descriptor")
            N = len(some_descriptors[n])
            x = np.arange(N)
            plt.bar(x, some_descriptors[n], width=1.0)
            plt.axis('off')

    @staticmethod
    def show_corners_on_image(img, corners):
        """Shows the extracted corners on the image

        Args:
            img (np.array): image array
            corners (np.array): binary corner image
        """
        img_3channels = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_3channels[corners] = [255, 0, 0]
        plt.imshow(img_3channels)
