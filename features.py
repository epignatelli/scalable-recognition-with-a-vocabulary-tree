import numpy as np
import math
import time
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


class SIFTDescriptor(object):
    """Class for computing SIFT descriptor of the square patch

    From https://github.com/ducha-aiki/numpy-sift

    Attributes:
        patchSize: size of the patch in pixels
        maxBinValue: maximum descriptor element after L2 normalization. All above are clipped to this value
        numOrientationBins: number of orientation bins for histogram
        numSpatialBins: number of spatial bins. The final descriptor size is numSpatialBins x numSpatialBins x numOrientationBins
    """

    def precomputebins(self):
        halfSize = int(self.patchSize/2)
        ps = self.patchSize
        sb = self.spatialBins
        step = float(self.spatialBins + 1) / (2 * halfSize)
        precomp_bins = np.zeros(2 * ps, dtype=np.int32)
        precomp_weights = np.zeros(2*ps, dtype=np.float)
        precomp_bin_weights_by_bx_py_px_mapping = np.zeros(
            (sb, sb, ps, ps), dtype=np.float)
        for i in range(ps):
            i1 = i + ps
            x = step * i
            xi = int(x)
            # bin indices
            precomp_bins[i] = xi - 1
            precomp_bins[i1] = xi
            # bin weights
            precomp_weights[i1] = x - xi
            precomp_weights[i] = 1.0 - precomp_weights[i1]
            # truncate
            if (precomp_bins[i] < 0):
                precomp_bins[i] = 0
                precomp_weights[i] = 0
            if (precomp_bins[i] >= self.spatialBins):
                precomp_bins[i] = self.spatialBins - 1
                precomp_weights[i] = 0
            if (precomp_bins[i1] < 0):
                precomp_bins[i1] = 0
                precomp_weights[i1] = 0
            if (precomp_bins[i1] >= self.spatialBins):
                precomp_bins[i1] = self.spatialBins - 1
                precomp_weights[i1] = 0
        for y in range(ps):
            for x in range(ps):
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y],
                                                        precomp_bins[x], y, x] += precomp_weights[y]*precomp_weights[x]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y+ps],
                                                        precomp_bins[x], y, x] += precomp_weights[y+ps]*precomp_weights[x]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y],
                                                        precomp_bins[x+ps], y, x] += precomp_weights[y]*precomp_weights[x+ps]
                precomp_bin_weights_by_bx_py_px_mapping[precomp_bins[y+ps],
                                                        precomp_bins[x+ps], y, x] += precomp_weights[y+ps]*precomp_weights[x+ps]
        if self.mask_type == 'CircularGauss':
            mask = self.CircularGaussKernel(
                kernlen=self.patchSize, circ=True, sigma_type=self.sigma_type).astype(np.float32)
        elif self.mask_type == 'Gauss':
            mask = self.CircularGaussKernel(
                kernlen=self.patchSize, circ=False, sigma_type=self.sigma_type).astype(np.float32)
        else:
            raise ValueError(self.mask_type, 'is unknown mask type')

        for y in range(sb):
            for x in range(sb):
                precomp_bin_weights_by_bx_py_px_mapping[y, x, :, :] *= mask
                precomp_bin_weights_by_bx_py_px_mapping[y, x, :, :] = np.maximum(
                    0, precomp_bin_weights_by_bx_py_px_mapping[y, x, :, :])
        return precomp_bins.astype(np.int32), precomp_weights, precomp_bin_weights_by_bx_py_px_mapping, mask

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'orientationBins=' + str(self.orientationBins) +\
            ', ' + 'spatialBins=' + str(self.spatialBins) +\
            ', ' + 'patchSize=' + str(self.patchSize) +\
            ', ' + 'sigma_type=' + str(self.sigma_type) +\
            ', ' + 'mask_type=' + str(self.mask_type) +\
            ', ' + 'maxBinValue=' + str(self.maxBinValue) + ')'

    def __init__(self, patchSize=41,
                 maxBinValue=0.2,
                 numOrientationBins=8,
                 numSpatialBins=4,
                 mask_type='CircularGauss',
                 sigma_type='hesamp'):
        self.patchSize = patchSize
        self.maxBinValue = maxBinValue
        self.orientationBins = numOrientationBins
        self.spatialBins = numSpatialBins
        self.mask_type = mask_type
        self.sigma_type = sigma_type
        self.precomp_bins, self.precomp_weights, self.mapping, self.mask = self.precomputebins()
        self.binaryMask = self.mask > 0
        self.gx = np.zeros((patchSize, patchSize), dtype=np.float)
        self.gy = np.zeros((patchSize, patchSize), dtype=np.float)
        self.ori = np.zeros((patchSize, patchSize), dtype=np.float)
        self.mag = np.zeros((patchSize, patchSize), dtype=np.float)
        self.norm_patch = np.zeros((patchSize, patchSize), dtype=np.float)
        sb = self.spatialBins
        ob = self.orientationBins
        self.desc = np.zeros((ob, sb, sb), dtype=np.float)
        return

    def CircularGaussKernel(self, kernlen=21, circ=True, sigma_type='hesamp'):
        halfSize = float(kernlen) / 2.
        r2 = float(halfSize**2)
        if sigma_type == 'hesamp':
            sigma_mul_2 = 0.9 * r2
        elif sigma_type == 'vlfeat':
            sigma_mul_2 = kernlen**2
        else:
            raise ValueError('Unknown sigma_type', sigma_type,
                             'try hesamp or vlfeat')
        disq = 0
        kernel = np.zeros((kernlen, kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize+0.5)**2 + (x - halfSize+0.5)**2
                kernel[y, x] = math.exp(-disq / sigma_mul_2)
                if circ and (disq >= r2):
                    kernel[y, x] = 0.
        return kernel

    def photonorm(self, patch, binaryMask=None):
        if binaryMask is not None:
            std1_coef = 50. / np.std(patch[binaryMask])
            mean1 = np.mean(patch[binaryMask])
        else:
            std1_coef = 50. / np.std(patch)
            mean1 = np.mean(patch)
        if std1_coef >= 50. / 0.000001:
            std1_coef = 50.0
        self.norm_patch = 128. + std1_coef * (patch - mean1)
        self.norm_patch = np.clip(self.norm_patch, 0., 255.)
        return

    def getDerivatives(self, image):
        # [-1 1] kernel for borders
        self.gx[:, 0] = image[:, 1] - image[:, 0]
        self.gy[0, :] = image[1, :] - image[0, :]
        self.gx[:, -1] = image[:, -1] - image[:, -2]
        self.gy[-1, :] = image[-1, :] - image[-2, :]
        # [-1 0 1] kernel for the rest
        self.gy[1:-2, :] = image[2:-1, :] - image[0:-3, :]
        self.gx[:, 1:-2] = image[:, 2:-1] - image[:, 0:-3]
        self.gx *= 0.5
        self.gy *= 0.5
        return

    def samplePatch(self, grad, ori):
        ps = self.patchSize
        sb = self.spatialBins
        ob = self.orientationBins
        o_big = float(ob) * (ori + 2.0*math.pi) / (2.0 * math.pi)
        bo0_big = np.floor(o_big)  # .astype(np.int32)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % ob
        bo1_big = (bo0_big + 1.0) % ob
        wo0_big = 1.0 - wo1_big
        wo0_big *= grad
        wo0_big = np.maximum(0, wo0_big)
        wo1_big *= grad
        wo1_big = np.maximum(0, wo1_big)
        ori_weight_map = np.zeros((ob, ps, ps))
        for o in range(ob):
            relevant0 = np.where(bo0_big == o)
            ori_weight_map[o, relevant0[0], relevant0[1]
                           ] = wo0_big[relevant0[0], relevant0[1]]
            relevant1 = np.where(bo1_big == o)
            ori_weight_map[o, relevant1[0], relevant1[1]
                           ] += wo1_big[relevant1[0], relevant1[1]]
        for y in range(sb):
            for x in range(sb):
                self.desc[:, y, x] = np.tensordot(
                    ori_weight_map, self.mapping[y, x, :, :])
        return

    def describe(self, patch, userootsift=False, flatten=True, show_timings=False):
        t = time.time()
        self.photonorm(patch, binaryMask=self.binaryMask)
        if show_timings:
            print('photonorm time = ', time.time() - t)
            t = time.time()
        self.getDerivatives(self.norm_patch)
        if show_timings:
            print('gradients time = ', time.time() - t)
            t = time.time()
        self.mag = np.sqrt(self.gx * self.gx + self.gy*self.gy)
        self.ori = np.arctan2(self.gy, self.gx)
        if show_timings:
            print('mag + ori time = ', time.time() - t)
            t = time.time()
        self.samplePatch(self.mag, self.ori)
        if show_timings:
            print('sample patch time = ', time.time() - t)
            t = time.time()
        self.desc /= np.linalg.norm(self.desc.flatten(), 2)
        self.desc = np.clip(self.desc, 0, self.maxBinValue)
        self.desc /= np.linalg.norm(self.desc.flatten(), 2)
        if userootsift:
            self.desc = np.sqrt(
                self.desc / np.linalg.norm(self.desc.flatten(), 1))
        if show_timings:
            print('clip and norm time = ', time.time() - t)
            t = time.time()
        if flatten:
            return np.clip(512. * self.desc.flatten(), 0, 255).astype(np.int32)
        else:
            return np.clip(512. * self.desc, 0, 255).astype(np.int32)


def extract_MSER_patches(img, blobs, square=False):
    """Extracts the patches associated to the keypoints of the MSER
    descriptors.

    Arguments:
        img {np.array} -- Image from which the patches are extracted
        blobs -- Bounding boxes as returned by the function mser.detectRegions()

    Returns:
        [np.array] -- List of patches
    """
    patch_size = (64, 64)
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
        resized = cv2.resize(img_crop, patch_size,
                             interpolation=cv2.INTER_LANCZOS4)

        patches.append(resized)
    return patches


def show_MSER_blobs(img, blobs):
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
    show_image(canvas1)
    plt.subplot(122)
    show_image(canvas3)


# Extracts the image patch given the keypoint
def patch_from_keypoint(img, keypoint):
    patch_center = np.array(keypoint.pt).astype(np.int)
    patch_size = int(keypoint.size/1.5)
    angle = keypoint.angle

    # Extracting large patch around center
    patch_x = int(patch_center[1] - patch_size)
    patch_y = int(patch_center[0] - patch_size)
    x0 = np.amax([0, patch_x])
    y0 = np.amax([0, patch_y])
    x1 = np.amin([img.shape[0], patch_x + 2 * patch_size])
    y1 = np.amin([img.shape[1], patch_y + 2 * patch_size])
    patch_image = img[x0:x1, y0:y1]

    # Rotating patch and cropping (This does nothing without SIFT)
    rows, cols, = patch_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    patch_image = cv2.warpAffine(patch_image, M, (cols, rows))
    return patch_image


# Shows N descriptors taken at random
def show_random_descriptors(img, patches, descriptors, N=5):
    # Getting random keypoints
    random_idx = [random.randint(0, len(patches) - 1) for n in range(N)]
    some_patches = [patches[i] for i in random_idx]
    some_descriptors = [descriptors[i] for i in random_idx]

    # Setting up axes
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    gs = GridSpec(N, 8, figure=fig)

    ax1 = fig.add_subplot(gs[:, :-2])
    ax1.set_title("Image")
    show_image(img)
    plt.axis('off')

    # Showing the patched with their desctiptors
    for n in range(N):
        # Getting and showing patch
        patch = some_patches[n]
        ax = fig.add_subplot(gs[n, -2])
        if n == 0:
            ax.set_title("Patch")
        show_image(patch)
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


def show_image(img, gray=False, image_size=6):
    if not gray:
        plt.imshow(img, aspect="equal")
    else:
        plt.imshow(img, aspect="equal", cmap="gray")


def show_corners_on_image(img, corners):
    img_3channels = cv2.cvtColor(img / 255, cv2.COLOR_GRAY2RGB)
    img_3channels[corners] = [1, 0, 0]
    show_image(img_3channels)


# Read image and change the color space
def detect_MSER_blobs(img, min_area=4000, max_area=200000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get mser, and set parameters
    mser = cv2.MSER_create(_max_variation=0.5)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)

    # Do mser detection, get the coodinates and bboxes
    coordinates, bboxes = mser.detectRegions(gray)

    # Filter the coordinates
    coords = []
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x, y, w, h = bbox
        if w < 20 or h < 20 or w / h > 5 or h / w > 5:
            continue
        coords.append(coord)
    return coords
