import os
import numpy as np
import matplotlib.pyplot as plt
from .dataset import Dataset
import pickle
from . import utils


class Database(object):
    def __init__(self, dataset, encoder):
        # public:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, str):
            self.dataset = Dataset(dataset)
        else:
            raise TypeError("Invalid dataset of type %s" % type(dataset))
        self.encoder = encoder

        # private:
        self._database = {}
        self._image_ids = {}
        return

    def get_image_id(self, image_path):
        # normalise the path first
        image_path = os.path.abspath(image_path)

        # lookup if image has already been hahed
        if image_path not in self._image_ids:
            # otherwise, store it
            self._image_ids[image_path] = hash(image_path)

        return self._image_ids[image_path]

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("\nGenerating index...")
        utils.show_progress(self.embedding, self.dataset.image_paths)
        return

    def is_indexed(self, image_path):
        image_id = self.get_image_id(image_path)
        return image_id in self._database

    def embedding(self, image_path):
        image_id = self.get_image_id(image_path)
        # check if has already been indexed
        if image_id not in self._database:
            # if not, calculate the embedding and index it
            image = self.dataset.read_image(image_path)
            self._database[image_id] = self.encoder.embedding(image)
        return self._database[image_id]

    def score(self, db_image_path, query_image_path):
        """
        Measures the similatiries between the set of paths of the features of each image.
        """
        # get the vectors of the images
        d = self.embedding(db_image_path)
        q = self.embedding(query_image_path)
        d = d / np.linalg.norm(d, ord=2)
        q = q / np.linalg.norm(q, ord=2)
        # simplified scoring using the l2 norm
        score = np.linalg.norm(d - q, ord=2)
        return score if not np.isnan(score) else 1e6

    def retrieve(self, query_image_path, n=4):
        # propagate the query down the tree
        scores = {}
        for db_image_path in self.dataset.image_paths:
            scores[db_image_path] = self.score(db_image_path, query_image_path)

        # sorting scores
        sorted_scores = {k: v for k, v in sorted(
            scores.items(), key=lambda item: item[1])}
        return sorted_scores

    def save(self, path=None):
        if path is None:
            path = "data"

        # store indexed vectors in hdf5
        with open(os.path.join(path, "index.pickle"), "wb") as f:
            pickle.dump(self._database, f)

        return True

    def load(self, path="data"):
        # load indexed vectors from pickle
        try:
            with open(os.path.join(path, "index.pickle"), "rb") as f:
                database = pickle.load(f)
                self._database = database
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
            ax[i].imshow(self.dataset.read_image(img_ids[i]))
            ax[i].set_title("#%d. %s Score:%.3f" %
                            (i, img_ids[i], scores[i]))
        return
