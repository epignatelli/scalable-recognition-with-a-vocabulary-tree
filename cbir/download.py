import urllib.request
import os
from multiprocessing.pool import ThreadPool
import tarfile

INRIA_DESCRIPTORS = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/siftgeo.tar.gz", 538744, 551669)  # url, size on ftp, size on disk
INRIA_IMAGES1 = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz", 1115072, 1141827)  # url, size on ftp, size on disk
INRIA_IMAGES2 = ("ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz", 1661496, 1701364)  # url, size on ftp, size on disk
TOTAL_SIZE = (INRIA_DESCRIPTORS[1] + INRIA_IMAGES1[1] + INRIA_IMAGES2[1]) / 1024
DOWNLOAD_INTO = "data"

downloaded = 0
total_to_extract = 0
extracted = 0


def show_download_progress(block_num, block_size, total_size):
    global downloaded
    downloaded += block_size / 1024**2
    print(f"Downloading data: {int(downloaded)} MB/{TOTAL_SIZE} MB", end="\r")

    
def download(file):
    os.makedirs("data", exist_ok=True)
    url, size, size_on_disk = file
    out_path = os.path.join(DOWNLOAD_INTO, os.path.basename(url))
    if (is_downloaded(file)):
        return
    urllib.request.urlretrieve(url, out_path, show_download_progress)

    
def download_parallel():
    #with ThreadPool(3) as p:
    #    p.map(download, [INRIA_DESCRIPTORS, INRIA_IMAGES1, INRIA_IMAGES2])
    download(INRIA_DESCRIPTORS)
    download(INRIA_IMAGES1)
    download(INRIA_IMAGES2)
    print("\nData downloaded.")

    
def is_downloaded(file):
    url, size, size_on_disk = file
    path = os.path.join(DOWNLOAD_INTO, os.path.basename(url))
    if not (os.path.exists(path)):
        return False
    if (os.path.getsize(path) / 1024 < size_on_disk * 0.9):  # at least 90%, since sizes are not precise
        return False
    print(f"{path} already downloaded in {os.path.abspath(path)}")
    return True


def extract(file):
    global total_to_extract
    global extracted
    if (is_extracted(file)):
        return
    url, size, size_on_disk = file
    filepath = os.path.join(DOWNLOAD_INTO, os.path.basename(url))
    if not (os.path.exists(filepath)):
        print (f"File {filepath} not found on disk.")
        return
    with tarfile.open(filepath, "r:gz") as tar:
        members = tar.getmembers()
        total_to_extract += len(members)
        for member in members:
            extracted += 1
            print(f"Extracting {extracted}/{total_to_extract}", end="\r")
            tar.extract(path="data", member=member)

            
def extract_parallel():
    with ThreadPool(3) as p:
        print("Inflating compressed archives")
        p.map(extract, [INRIA_DESCRIPTORS, INRIA_IMAGES1, INRIA_IMAGES2])
    print("\nExtraction completed.")

    
def is_extracted(file):
    url, size, size_on_disk = file
    filepath = os.path.join("data", os.path.basename(url))
    dirname = os.path.splitext(os.path.splitext(filepath)[0])[0]
    dirname = ''.join([i for i in dirname if not i.isdigit()])  # remove numbers from string
    if not (os.path.isdir(dirname)):
        return False
    print(f"{filepath} already extracted in {os.path.abspath(dirname)}")
    return True


if __name__ == "__main__":
    download_parallel()
    extract_parallel()
