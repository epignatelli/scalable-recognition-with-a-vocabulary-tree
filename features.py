import numpy as np
import cv2
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pytorch_sift.pytorch_sift import SIFTNet


def Descriptor(object):
    def __init__(self, patch_size=65, mser_min_area=4000, mser_max_area=200000):
        # this sets self.describe to the SIFTNet callable
        self.patch_size = (int(patch_size), int(patch_size))
        self.sift = SIFTNet(patch_size=patch_size,
                            sigma_type="vlfeat", masktype='Gauss')

        # Creating the detector and setting some properties
        # see --> https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html
        self.mser = cv2.MSER_create(
            _max_variation=0.5, _min_area=mser_min_area, _max_area=mser_max_area)

    def find_keypoints(self, image):
        # Making the images grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detecting the keypoints
        kp = self.mser.detect(image)

        # Getting the mser regions for drawing Bounding-Box
        blobs, bboxes = self.mser.detectRegions(gray)
        return kp, blobs

    def extract_patches(self, img, blobs, square=False):
        """Extracts the patches associated to the keypoints of the MSER
        descriptors.

        Arguments:
            img {np.array} -- Image from which the patches are extracted
            blobs -- Bounding boxes as returned by the function mser.detectRegions()

        Returns:
            [np.array] -- List of patches
        """
        patches = []
        for cnt in blobs:
            rect = cv2.minAreaRect(cnt)
            # rotate img
            angle = rect[2]
            rows, cols = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            img_rot = cv2.warpAffine(img, M, (cols, rows))

            # rotate bounding box
            rect0 = (rect[0], rect[1], 0.0)
            box = cv2.boxPoints(rect0)
            pts = np.int0(cv2.transform(np.array([box]), M))[0]
            pts[pts < 0] = 0

            # crop
            img_crop = img_rot[pts[1][1]:pts[0][1],
                               pts[1][0]:pts[2][0]]

            if img_crop.size == 0:
                continue

            # Squares the image
            resized = cv2.resize(img_crop, self.patch_size,
                                 interpolation=cv2.INTER_LANCZOS4)

            patches.append(resized)
        return patches

    def describe(self, patch):
        """
        Computes the SIFT descriptor on the given path.
        Note that we implement vlfeat version of sift
        """
        return self.sift(torch.as_tensor(patch))

    def show_blobs(img, blobs):
        canvas1 = img.copy()
        canvas3 = np.zeros_like(img)
        for cnt in blobs:
            # Show in separate image
            xx = cnt[:, 0]
            yy = cnt[:, 1]
            color = [random.randint(0, 255) for _ in range(3)]
            canvas3[yy, xx] = color

            # Show as BBox
            # get the min area rect
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            canvas1 = cv2.drawContours(canvas1, [box], 0, (0, 255, 0), 1)

        # Show
        plt.subplot(121)
        plt.imshow(canvas1)
        plt.subplot(122)
        plt.imshow(canvas3)

    def show_random_descriptors(img, patches, descriptors, N=5):
        """
        Shows N descriptors taken at random
        """
        # Getting random keypoints
        random_idx = [random.randint(0, len(patches) - 1) for n in range(N)]
        some_patches = [patches[i] for i in random_idx]
        some_descriptors = [descriptors[i] for i in random_idx]

        # Setting up axes
        fig = plt.figure(constrained_layout=True, figsize=(15, 8))
        gs = GridSpec(N, 8, figure=fig)

        ax1 = fig.add_subplot(gs[:, :-2])
        ax1.set_title("Image")
        plt.imshow(img)
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
            clist = [(0, "#c58882"), (1, "#1d201f")]
            rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
            if n == 0:
                ax.set_title("Descriptor")
            N = len(some_descriptors[n])
            x = np.arange(N)
            plt.bar(x, some_descriptors[n], color=rvb(x / N), width=1.0)
            plt.axis('off')

    def show_corners_on_image(img, corners):
        img_3channels = cv2.cvtColor(img / 255, cv2.COLOR_GRAY2RGB)
        img_3channels[corners] = [1, 0, 0]
        plt.imshow(img_3channels)
