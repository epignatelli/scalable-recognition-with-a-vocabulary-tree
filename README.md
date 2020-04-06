## Scalable Recognition with a Vocabulary Tree, D. Nister, H. Stewenius, 2006
#### Abstract

> A recognition scheme that scales efficiently to a large number of objects is presented. The efficiency and quality is exhibited in a live demonstration that recognizes CD-covers from a database of 40000 images of popular music CD’s. The scheme builds upon popular techniques of indexing descriptors extracted from local regions, and is robust to background clutter and occlusion. The local region descriptors are hierarchically quantized in a vocabulary tree. The vocabulary tree allows a larger and more discriminatory vocabulary to be used efficiently, which we show experimentally leads to a dramatic improvement in retrieval quality. The most significant property of the scheme is that the tree directly defines the quantization. The quantization and the indexing are therefore fully integrated, essentially being one and the same. The recognition quality is evaluated through retrieval on a database with ground truth, showing the power of the vocabulary tree approach, going as high as 1 million images.


## Getting started

This repository provide a python implementation of the paper above.

#### 1. Install the conda environment
   A. If you are on windows
```
conda env create -f env\sberbank_win.yaml
```

   B. If you are on macOS or on linux platforms

```
conda env create -f env\sberbank_unix.yaml
```

#### 2. Start jupyter and open the notebook
```
conda activate sberbank
jupyter-notebook
```

#### 3. Open a terminal from jupyter and type
```
python cbir/download.py
```

## Example

```python
import cbir
import random

# create the dataset
# for the sake of speed, we will do it in a subset
root = "path to a folder that contains a list of images"
dataset = cbir.Dataset(root)
subset = dataset.subset[0:10]

# try plotting some of the images
image_path = random.choice(subset)
subset.show_image(image_path)

# create the vocabulary tree

orb = cbir.descriptors.Orb()
voc = cbir.encoders.VocabularyTree(n_branches=3, depth=3, descriptor=orb)
voc.learn(subset)

# and now create the database
db = cbir.Database(subset, encoder=voc)

# let's generate the index
db.index()

# and test a retrieval
query_path = "100000.jpeg"
scores = db.retrieve(query_path)
db.show_results(query_path, scores)
```

You can easily change the descriptor or the encoder to improve your results.

An example using the probabilities of an `AlexNet` as embedding
```python
db = cbir.Database(subset, encoder=cbir.encoders.AlexNet())
db.index()

# retrieval
query_path = "100000.jpeg"
scores = db.retrieve(query_path)
db.show_results(query_path, scores)
```

An example using the last layer of AlexNet as descriptors for the vocabulary tree
```python
voc = cbir.encoders.VocabularyTree(n_branches=3, depth=3, descriptor=cbir.descriptors.AlexNet())
voc.learn(subset)

# database
db = cbir.Database(subset, encoder=voc)
db.index()

# retrieval
query_path = "100000.jpeg"
scores = db.retrieve(query_path)
db.show_results(query_path, scores)
```

## Performance test

```python
import cbir

dataset = cbir.Dataset().subset[0:100]

orb = cbir.descriptors.Orb()
voc = cbir.encoders.VocabularyTree(n_branches=4, depth=4, descriptor=orb)

features = voc.extract_features(dataset)

%time voc.fit(features)
```
```
CPU times: user 1min 43s, sys: 3.17 s, total: 1min 46sde 336 at level 3			
Wall time: 56.6 s
```
```python
db = cbir.Database(dataset, encoder=voc)
%time db.index()
```
```
CPU times: user 9min 2s, sys: 3.85 s, total: 9min 6s
Wall time: 8min 46s
```
```python
import random
query = random.choice(dataset)
%time scores = db.retrieve(query)
```
```
CPU times: user 73.1 ms, sys: 7 µs, total: 73.1 ms
Wall time: 72.5 ms
```

## [Dev] Add new descriptors or encoders
Do add your own descriptors and encoders and tell us how they've done!

To add a new descriptor:
```python
from . import DescriptorBase
class NewDescriptor(DescriptorBase):
   def describe(self, image_cv):
      # do stuff
      return the descriptor
```

To add a new encoder:
```python
class NewEncoder(object):
   def embedding(self, image_cv):
      # do stuff
      return the embedding
```

## Literature

#### Datasets:
- [Hamming embedding and weak geometricconsistency for large scale image search (INRIA Holydays)](http://lear.inrialpes.fr/people/jegou/data.php#holidays) - [download](https://lear.inrialpes.fr/pubs/2008/JDS08/jegou_hewgc08.pdf)
- [Large-scale Landmark Retrieval/Recognition under a Noisy and Diverse Dataset](https://arxiv.org/abs/1906.04087)
- [INSTRE: a New Benchmark for Instance-Level Object Retrieval and Recognition](https://dl.acm.org/doi/pdf/10.1145/2700292)

#### Database indexing:
- [PQk-means: Billion-scale Clustering forProduct-quantized Codes](https://arxiv.org/pdf/1709.03708.pdf)
- [Scalable Recognition with a Vocabulary Tree](https://ieeexplore.ieee.org/document/1641018)
- [Object retrieval with large vocabularies and fast spatial matching](https://ieeexplore.ieee.org/document/4270197)

#### Features extraction:
- [Neural Codes for Image Retrieval](https://arxiv.org/pdf/1404.1777.pdf)
- [Scale-Invariant Feature Transform](https://pdfs.semanticscholar.org/0129/3b985b17154fbb178cd1f944ce3cc4fc9266.pdf)
- [Slides on SIFT (Lect. 11 and 12)](http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/)
- [Object Recognition from Local Scale-Invariant Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=790410)
- [Distinctive Image Features from Scale-Invariant Keypoints](https://link.springer.com/content/pdf/10.1023/B:VISI.0000029664.99615.94.pdf)
- [Speeded Up Ro- bust Feature (SURF)](https://www.vision.ee.ethz.ch/~surf/eccv06.pdf)
- [Using very deep autoencoders for content-based image retrieval](http://www.cs.toronto.edu/~fritz/absps/esann-deep-final.pdf)
- [Video Google: A Text Retrieval Approach to Object Matching in Videos](http://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf)

#### End-to-end
- [Large Scale Online Learning of Image Similarity Through Ranking (Triplet loss)](http://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf)
- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)

#### Surveys
- [A survey on Image Retrieval Methods](http://cogprints.org/9815/1/Survey%20on%20Image%20Retrieval%20Methods.pdf)
- [Recent Advance in Content-based ImageRetrieval: A Literature Survey](https://arxiv.org/pdf/1706.06064.pdf)

#### Code
- https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb
- https://github.com/ducha-aiki/pytorch-sift
