import subprocess
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
from .descriptor_base import DescriptorBase


class EzSIFT(DescriptorBase):
    def __init__(self):
        super(EzSIFT, self).__init__("data")
        if os.name == "nt":
            self.program = "./ezsift_win"
        elif os.name == "posix":
            self.program = "./ezsift_unix"
        return

    def __call__(self, img):
        return self.describe(img)

    def describe(self, img):
        if not os.path.exists('data'):
            os.makedirs('data')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('data/do_no_delete_me.pgm', img)
        cv2.imwrite('data/do_no_delete_me.jpg', img)

        p = subprocess.Popen([self.program, 'data/do_no_delete_me.pgm'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate()

        df = self.read_keypoints_file()
        if df is None:
            return []

        KPs, descriptors = self.dataframe_to_sift(df)

        # ToDo - sanity check on number of descriptors
        return descriptors

    def read_keypoints_file(self):
        path = "data/do_no_delete_me.pgm_sift_key.key"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, skiprows=(1), sep='\t',
                                 engine='python', header=None)
                # os.remove(path)
                return df
            except:
                warnings.warn("Features extraction failed for current image")
                return None
        else:
            raise FileNotFoundError(os.path.abspath(path))

    def dataframe_to_sift(self, df):
        """
        in:      Descriptors within a dataframe, df
        returns: dict array of KPs and NKPs x 128 array of descriptors
        """
        KPsDF = df.iloc[:, 2:6]
        DesDF = df.iloc[:, 6:134]

        NKPs = KPsDF.shape[0]

        KPs = []
        Descriptors = np.ndarray(shape=(NKPs, 128), dtype=int)

        for index, row in KPsDF.iterrows():
            KP = {'Location': (row.iloc[0], row.iloc[1]),
                  'Scale': row.iloc[2],
                  'Angle': row.iloc[3]
                  }
            KPs.append(KP)

            Descriptors[index, :] = DesDF.iloc[index]

        return KPs, Descriptors

    def show_keypoints(self, img1, KPs, KPsize=2, color='r'):
        plt.imshow(img1, cmap=plt.get_cmap('Greys_r'))
        plt.colorbar()
        ax = plt.gca()

        for kp in KPs:
            y, x = kp['Location']
            r = 1.5*kp['Scale']
            ax.plot(x, y, 'ro', markersize=KPsize)

            ax.add_patch(Circle((x, y), radius=r, edgecolor=[
                         0.5, 0.5, 1], facecolor='none', linewidth=1))
