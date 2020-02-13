import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import random

class Dataset():
    def __init__(self, folder="jpg"):
        self.path = 'data/' + folder
        self.all_images = [f for f in listdir(self.path) if isfile(join(self.path, f))]
    
    def print_files(self):
        print(self.all_images[:5])
        
    def get_image_by_name(self, image_name=None, gray=True):
        image = cv2.imread(self.path + '/' + image_name)
        if gray:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return np.float32(gray)
        else:
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    def get_random_image(self, gray=False):
        return self.get_image_by_name(random.choice(self.all_images),gray)

def show_image(img, gray=False, image_size=6):
    if not gray:
        plt.imshow(img,aspect="equal")
    else:
        plt.imshow(img,aspect="equal", cmap="gray")
    
def show_corners_on_image(img, corners):
    img_3channels = cv2.cvtColor(img/255,cv2.COLOR_GRAY2RGB)
    img_3channels[corners]=[1,0,0]
    show_image(img_3channels)

## Read image and change the color space
def detect_MSER_blobs(img, min_area=4000, max_area=200000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Get mser, and set parameters
    mser = cv2.MSER_create(_max_variation = 0.5)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    
    ## Do mser detection, get the coodinates and bboxes
    coordinates, bboxes = mser.detectRegions(gray)

    ## Filter the coordinates
    vis = img.copy()
    coords = []
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x,y,w,h = bbox
        if w< 20 or h < 20 or w/h > 5 or h/w > 5:
            continue
        coords.append(coord)
    return coords

def show_MSER_blobs(img, blobs):

    ## colors 

    ## Fill with random colors
    np.random.seed(0)
    canvas1 = img.copy()
    canvas3 = np.zeros_like(img)

    for cnt in blobs:
        # Show in separate image
        xx = cnt[:,0]
        yy = cnt[:,1]
        color = [random.randint(0, 255) for _ in range(3)]
        canvas3[yy, xx] = color

        # Show as BBox
        # get the min area rect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        canvas1 = cv2.drawContours(canvas1, [box], 0, (0, 255, 0), 3)

    ## Show
    RGB_image = cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    show_image(RGB_image)
    plt.subplot(122)
    show_image(canvas3)