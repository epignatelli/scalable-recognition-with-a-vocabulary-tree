import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset
import pickle
import utils


class CBIR(object):
    def __init__(self, root, encoder):
        self.dataset = Dataset(root)
        self.encoder = encoder
        self.database = {}

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("\nGenerating index...")
        utils.show_progress(self.embedding, self.dataset.all_images)
        return

    def is_indexed(self, image_id):
        return image_id in self.database

    def embedding(self, image_id):
        if not self.is_indexed(image_id):
            image = self.dataset.read_image(image_id)
            self.database[image_id] = self.encoder.embedding(image)
        return self.database[image_id]

    def score(self, db_id, query_id):
        """
        Measures the similatiries between the set of paths of the features of each image.
        """
        # get the vectors of the images
        d = self.embedding(db_id, return_graph=False)
        q = self.embedding(query_id, return_graph=False)
        d = d / np.linalg.norm(d)
        q = q / np.linalg.norm(q)
        # simplified scoring using the l2 norm
        score = np.linalg.norm(d - q, ord=2)
        return score if not np.isnan(score) else 1e6

    def retrieve(self, query_id, n=4):
        # propagate the query down the tree
        scores = {}
        for database_image_path in self.dataset.all_images:
            db_id = utils.get_image_id(database_image_path)
            scores[db_id] = self.score(database_image, query_image)
        sorted_scores = {k: v for k, v in sorted(
            scores.items(), key=lambda item: item[1])}
        return sorted_scores

    def save(self, path=None):
        if path is None:
            path = "data"

        # store indexed vectors in hdf5
        with open(os.path.join(path, "index.pickle"), "wb") as f:
            pickle.dump(self.database, f)

        return True

    def load(self, path="data"):
        # load indexed vectors from pickle
        try:
            with open(os.path.join(path, "database.pickle"), "rb") as f:
                database = pickle.load(f)
                self.database = database
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
