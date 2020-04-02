import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataset import Dataset
import time
import warnings
import multiprocessing
import h5py
import pickle
import utils


class CBIR(object):
    def __init__(self, root, encoder, descriptor=None):
        self.dataset = Dataset(root)
        self.descriptor = descriptor
        self.encoder = encoder
        self.database = {}

    def extract_features(self, image=None):
        if self.descriptor is None:
            raise ValueError("You have not defined a features descriptor. "
                             "For a full list of descriptors, "
                             "check out the 'descriptors' namespace")

        if (image is not None):
            return self.descriptor.describe(image)
        print("Extracting features...")
        features = utils.show_progress(self.descriptor.describe, self.dataset.all_images)
        # features = []
        # times = []
        # total = len(self.dataset.all_images)
        # for i, path in enumerate(self.dataset.all_images):
        #     start = time.time()
        #     features.extend(self.descriptor.describe(path))
        #     times.append(time.time() - start)
        #     avg = np.mean(times)
        #     eta = avg * total - avg * (i + 1)
        #     print("Extracting features %d/%d from image %s - ETA: %2fs" % (i + 1, total, path, eta), end="\r")
        print("\n%d features extracted" % len(features))
        return np.array(features)

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("\nGenerating index...")
        embed = lambda id, db: db.update("id", self.encode(id)) if self.is_encoded(id)
        utils.show_progress(embed, self.dataset.all_images, db=self.database)

        # times = []
        # total = len(self.dataset.all_images)
        # done = 0
        # for i, image_path in enumerate(self.dataset.all_images):
        #     start = time.time()
        #     image_id = self.dataset.get_image_id(image_path)
        #     # if the imaged is already indexed, return
        #     if not self.is_encoded(image_id):
        #         embedding = self.encode(image_id)
        #         self.datbase[image_id] = embedding
        #     times.append(time.time() - start)
        #     avg = np.mean(times)
        #     eta = avg * total - avg * (i + 1)
        #     done += 1
        #     print("Indexing image %d/%d:  %s - ETA: %2fs" %
        #           (done + 1, total, image_path, eta), end="\r")

        # set weights of node based on entropy
        print("\nCalculating weights")
        N = len(self.dataset)
        for node_id, files in self.graph.nodes(data=True):
            N_i = len(files)
            if N_i:  # if the node is visited, calculate the weight, otherwise, leave it as initialised
                self.graph.nodes[node_id]["w"] = np.log(
                    N / N_i)  # calculate entropy
        print("Inverted index generated")
        return

    def is_indexed(self, image_id):
        return image_id in self.datbase

    def score(self, first_image_path, second_image_path):
        """
        Measures the similatiries between the set of paths of the features of each image.
        """
        # get the vectors of the images
        db_id = self.dataset.get_image_id(first_image_path)
        query_id = self.dataset.get_image_id(second_image_path)
        d = self.encode(db_id, return_graph=False)
        q = self.encode(query_id, return_graph=False)
        d = d / np.linalg.norm(d)
        q = q / np.linalg.norm(q)
        # simplified scoring using the l2 norm
        score = np.linalg.norm(d - q, ord=2)
        return score if not np.isnan(score) else 1e6

    def retrieve(self, query_image_path, n=4):
        # propagate the query down the tree
        self.propagate(query_image_path)

        scores = {}
        for database_image_path in self.dataset.all_images:
            db_id = self.dataset.get_image_id(database_image_path)
            scores[db_id] = self.score(database_image_path, query_image_path)
        sorted_scores = {k: v for k, v in sorted(
            scores.items(), key=lambda item: item[1])}
        return sorted_scores

    def save(self, path=None):
        if path is None:
            path = "data"

        # store indexed vectors in hdf5
        with open(os.path.join(path, "index.pickle"), "wb") as f:
            pickle.dump(self.datbase, f)

        return True

    def load(self, path=None):
        if path is None:
            path = "data"

        # load graph
        try:
            graph = nx.read_gpickle(os.path.join(path, "graph.pickle"))
            self.graph = graph
        except:
            print("Cannot read graph file at %s/graph.pickle" % path)

        # load nodes with features
        try:
            with open(os.path.join(path, "nodes.pickle"), "rb") as f:
                nodes = pickle.load(f)
                self.nodes = nodes
        except:
            print("Cannote read nodes file at %s/nodes.pickle" % path)

        # load indexed vectors from hdf5
        try:
            with open(os.path.join(path, "index.pickle"), "rb") as f:
                indexed = pickle.load(f)
                self.datbase = indexed
        except:
            print("Cannot load index file from %s/index.pickle" % path)
        return True

    def show_results(self, query_path, scores_dict, n=4, figsize=(10, 4)):
        fig, ax = plt.subplots(1, n + 1, figsize=figsize)
        ax[0].axis("off")
        ax[0].imshow(self.dataset.read_image(query_path))
        ax[0].set_title("Query image")
        img_ids = list(scores_dict.keys())
        scores = list(scores_dict.values())
        for i in range(1, len(ax)):
            ax[i].axis("off")
            ax[i].imshow(self.dataset.get_image_by_name(f"{img_ids[i]}.jpg"))
            ax[i].set_title("#%d. %s Score:%.3f" %
                            (i, img_ids[i], scores[i]))
        return
