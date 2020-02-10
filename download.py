from tqdm import tqdm_notebook as tqdm
import urllib.request
from ftplib import FTP
import os
from multiprocessing.pool import ThreadPool

INRIA_DESCRIPTORS_URL = "ftp://ftp.inrialpes.fr/pub/lear/douze/data/siftgeo.tar.gz"
INRIA_IMAGES1_URL = "ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz"
INRIA_IMAGES2_URL = "ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz"

def show_progress(block_num, block_size, total_size):
    print(f"{block_num * block_size // 1024 // 1024}Mb / {total_size}", end="\r")

def download(url):
    os.makedirs("data", exists_ok=True)
    out_path = os.path.join("data", os.path.basename(url))
    urllib.request.urlretrieve(url, out_path, show_progress)

def download_parallel():
    with ThreadPool(3) as p:
        p.map(download, [INRIA_DESCRIPTORS_URL, INRIA_IMAGES1_URL, INRIA_IMAGES2_URL])
    
if __name__ == "__main__":
    download_parallel()