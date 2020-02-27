import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from dataset import Dataset
import time


class CBIR(object):
    def __init__(self, root, n_branches, depth):
        self.dataset = Dataset(root)
        self.n_branches = n_branches
        self.depth = depth
        self.tree = {}
        self.nodes = {}

        # private:
        self._current_index = 0

        # init graph
        self.graph = nx.DiGraph()

        # build tree
        self.fit()
        self.draw()
        plt.show()
        # compute inverted index
        # self.index()
        return

    def extract_features(self, image=None):
        if (image is not None):
            return self.dataset.extract_features(image)
        else:
            features = []
            times = []
            total = len(self.dataset.all_images)
            for i, path in enumerate(self.dataset.all_images):
                start = time.time()
                features.extend(self.dataset.extract_features(path))
                times.append(time.time() - start)
                avg = np.mean(times)
                eta = avg * total - avg * (i + 1)
                print("Extracted features %d/%d from image %s - ETA: %ss" % (i + 1, total, path, eta), end="\r")
            print("\n%d features extracted" % len(features))
            return np.array(features)

    def fit(self, features=None, node=0, root=None, current_depth=0):
        """
        Generates a hierarchical vocabulary tree representation of some input features
        using hierarchical k-means clustering.
        This function populates two class fields:
            `self.tree`, as Dict[int, List[int]], where the key is the id of the root node
        and the value is a list of children nodes, and
             `self.nodes` as a dictionary Dict[int, numpy.ndarray] that stores the actual value for each node
        Args:
            features (numpy.ndarray): a two dimensional vector of input features where dim 0 is samples and dim 1 is features
            node (int): current node id to set
            root (numpy.ndarray): the value of the parent of the `node` as a virtual feature
            current_depth (int): the depth of the node as the distance in jumps from the very root of the tree
        """
        if features is None:
            features = self.extract_features()
        if root is None:
            root = np.mean(features)

        self.nodes[node] = root
        self.graph.add_node(node)

        # if `node` is a leaf node, return
        if current_depth >= self.depth or len(features) < self.n_branches:
            return

        # group features by cluster
        model = KMeans(n_clusters=self.n_branches)
        model.fit(features)
        children = [[] for i in range(self.n_branches)]
        for i in range(len(features)):
            children[model.labels_[i]].append(features[i])

        # cluster children
        self.tree[node] = []
        for i in range(self.n_branches):
            self._current_index += 1
            self.tree[node].append(self._current_index)
            self.graph.add_edge(node, self._current_index)
            self.fit(children[i], self._current_index,
                     model.cluster_centers_[i], current_depth + 1)
        return

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("Generating index")
        for image_path in self.dataset.all_images:
            self.propagate(image_path)

        # set weights of node based on entropy
        N = len(self.dataset)
        for node_id, files in self.graph.nodes(data=True):
            N_i = len(files)
            if N_i:  # if the node is visited, calculate the weight, otherwise, leave it as initialised
                self.graph.nodes[node_id]["w"] = np.log(
                    N / N_i)  # calculate entropy
        print("Inverted index generated")
        return

    def propagate(self, image_path):
        """
        Encodes an image into a set of paths on the tree.
        The vector representation is the values list of a key value pair,
        where the key is the id of the node and the value is the number of times that node is visited during propagation.
        This results into an tf-idf scheme.
        Args:
            image_path (str): path of the image to encode
        """
        print("Creating inverted index for %s" % image_path)
        features = self.extract_features(image_path)
        image_id = self.dataset.get_image_id(image_path)
        for feature in features:
            path = self.propagate_feature(feature)
            for i in range(len(path)):
                node = path[i]
                # add tfidf
                if image_id not in self.graph.nodes[node]:
                    self.graph.nodes[node][image_id] = 1
                else:
                    self.graph.nodes[node][image_id] += 1
        return

    def propagate_feature(self, feature, node=0):
        """
        Propagates a feature, down the tree, and returns the closest node.
        Args:
            feature (numpy.ndarray): The feature to lookup
            root (int): Node id to start the search from.
                        Default is 0, meaning the very root of the tree
        """
        min_dist = float("inf")
        path = [node]
        while self.graph.out_degree(node):  # stop if leaf
            print(node, self.graph.out_degree(node))
            for child in self.graph[node]:
                distance = np.linalg.norm([self.nodes[child] - feature])  # l1 norm
                if distance < min_dist:
                    # print(f"{node} to {child} is distant {distance}")
                    min_dist = distance
                    node = child
                    path.append(child)
        return path

    def encode(self, image_id, return_graph=True):
        subgraph = self.graph.subgraph(
            [k for k, v in self.graph.nodes(data=image_id, default=None) if v is not None])
        if return_graph:
            return subgraph
        weights = np.array(subgraph.nodes(data="w"))
        tfidf = np.array(subgraph.nodes(data=image_id))
        tfidf = tfidf / np.linalg.norm(tfidf)  # l2 norm
        return tfidf * weights

    def score(self, database_image_path, query_image_path):
        """
        Measures the similatiries between the set of paths of the features of each image.
        """
        db_id = self.dataset.get_image_id(database_image_path)
        query_id = self.dataset.get_image_id(query_image_path)

        # propagate the query down the tree
        self.propagate(query_image_path)

        # get the vectors of the images
        d = self.get_encoded(db_id, return_graph=False)
        q = self.get_encoded(query_id, return_graph=False)

        # simplified scoring using the l2 norm
        score = 2 - 2 * np.sum(d * q)
        return score

    def retrieve(self, query_image_path, n=4):
        scores = {}
        for database_image_path in self.dataset.all_images:
            db_id = self.dataset.get_image_id(database_image_path)
            scores[db_id] = self.score(database_image_path, query_image_path)
        sorted_scores = sorted(scores, key=scores.__getitem__)
        return sorted_scores.keys()[:n]

    def draw(self, figsize=None):
        figsize = (30, 10) if figsize is None else figsize
        plt.figure(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos=pos, with_labels=True)
