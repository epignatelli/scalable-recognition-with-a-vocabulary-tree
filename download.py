import urllib.request
import os
from multiprocessing.pool import ThreadPool

INRIA_DESCRIPTORS = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/siftgeo.tar.gz", 538744)
INRIA_IMAGES1 = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz", 1115072)
INRIA_IMAGES2 = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz", 1661496)
TOTAL_SIZE = INRIA_DESCRIPTORS[1] + INRIA_IMAGES1[1] + INRIA_IMAGES2[1]

downloaded = 0

def show_progress(block_num, block_size, total_size):
    global downloaded
    downloaded += block_size / 1024 / 1024
    print(f"Downloading data: {int(downloaded)} MB/{TOTAL_SIZE} MB", end="\r")

def download(file):
    os.makedirs("data", exist_ok=True)
    url, size = file
    out_path = os.path.join("data", os.path.basename(url))
    urllib.request.urlretrieve(url, out_path, show_progress)

def download_parallel():
    with ThreadPool(3) as p:
        p.map(download, [INRIA_DESCRIPTORS, INRIA_IMAGES1, INRIA_IMAGES2])
    
if __name__ == "__main__":
    download_parallel()
    print()
    print("Data downloaded.")