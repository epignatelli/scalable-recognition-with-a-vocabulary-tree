### Scalable Recognition with a Vocabulary Tree
The code provided in this repository has been developed for teaching purposes at the Imperial College London. 
It is part of the _Computer Vision Day_ of the Business School Executive Education Program for _Sberbank_.


## Getting started


#### 1. Install the conda environment
   A. If you are on windows
```
conda env create -f env\sberbank_win.yml
```

   B. If you are on macOS or on linux platforms

```
conda env create -f env\sberbank_unix.yml
```

#### 2. Start jupyter and open the notebook
```
conda activate cbir
jupyter lab
```

#### 3. Open a terminal from jupyter and type
```
python cbir/download.py
```


### Acknowledgements 
The authors acknowledge the Executive Education of the Business School at the Imperial College for the support.
We thank Professor Anil Bharath of the Department of Bioengineering for the guidance and the opportunity of being part of the _Computer Vision Day_.
Thanks to Kai Arulkumaran and to Stathi Fotiadis for the feedback before the session and the assistance in teaching the session (2020).


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
