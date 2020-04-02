import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataset import Dataset
import pickle
import utils


class CBIR(object):
    def __init__(self, root, encoder, descriptor=None):
        self.dataset = Dataset(root)
        self.descriptor = descriptor
        self.encoder = encoder
        self.database = {}

    def extract_features(self):
        if self.descriptor is None:
            raise ValueError("You have not defined a features descriptor. "
                             "For a full list of descriptors, "
                             "check out the 'descriptors' namespace")
        print("Extracting features...")
        features = utils.show_progress(
            self.descriptor.describe, self.dataset.all_images)
        print("\n%d features extracted" % len(features))
        return np.array(features)

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("\nGenerating index...")
        utils.show_progress(self.embedding, self.dataset.all_images)
        return

    def embedding(self, image_path):
        image_id = self.dataset.get_image_id(image_path)
        if not self.is_indexed(image_id):
            self.database[image_id] = self.encoder.embedding(image_id)
        return self.database[image_id]

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
            ax[i].imshow(self.dataset.get_image_by_name("%s.jpg" % img_ids[i]))
            ax[i].set_title("#%d. %s Score:%.3f" %
                            (i, img_ids[i], scores[i]))
        return
