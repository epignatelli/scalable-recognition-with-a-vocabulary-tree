import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import MiniBatchKMeans
from .. import utils


class VocabularyTree(object):
    def __init__(self, n_branches, depth, descriptor):
        self.n_branches = n_branches
        self.depth = depth
        self.descriptor = descriptor

        self.tree = {}
        self.nodes = {}
        self.graph = nx.DiGraph()

        # private:
        self._current_index = 0
        self._propagated = set()

    def learn(self, dataset):
        features = self.extract_features(dataset)
        self.fit(features)
        return self

    def extract_features(self, dataset):
        print("Extracting features...")
        func = lambda path: self.descriptor(dataset.read_image(path))
        features = utils.show_progress(func, dataset)
        print("\n%d features extracted" % len(features))
        return np.array(features)

    def fit(self, features, node=0, root=None, current_depth=0):
        """
        Generates a hierarchical vocabulary tree representation of some input features
        using hierarchical k-means clustering.
        This function populates the graph and stores the value of the features in
        `self.nodes` as a dictionary Dict[int, numpy.ndarray] that stores the actual value for each node
        Args:
            features (numpy.ndarray): a two dimensional vector of input features where dim 0 is samples and dim 1 is features
            node (int): current node id to set
            root (numpy.ndarray): the value of the parent of the `node` as a virtual feature
            current_depth (int): the depth of the node as the distance in jumps from the very root of the tree
        """
        if root is None:
            root = np.mean(features, axis=0)

        self.nodes[node] = root
        self.graph.add_node(node)

        # if `node` is a leaf node, return
        if current_depth >= self.depth or len(features) < self.n_branches:
            return

        # group features by cluster
        print("Computing clusters %d/%d with %d features from node %d at level %d\t\t" %
              (self._current_index, self.n_branches ** self.depth, len(features), node, current_depth),
              end="\r")
        model = MiniBatchKMeans(n_clusters=self.n_branches)
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

    def propagate(self, image):
        """
        Proapgates the features of an image down the tree, until the find a leaf.
        Every time they pass through a node, they leave a fingerprint, by storing a key value pairm
        where the key is the id of the image and the value is the number of times that node is visited.
        This results into an tf-idf scheme.
        Args:
            image_path (str): path of the image to encode
        """
        image_id = utils.get_image_id(image)
        if (image_id in self._propagated):
            return

        features = self.descriptor(image)
        for feature in features:
            path = self.propagate_feature(feature)
            for i in range(len(path)):
                node = path[i]
                # add tfidf
                if image_id not in self.graph.nodes[node]:
                    self.graph.nodes[node][image_id] = 1
                else:
                    self.graph.nodes[node][image_id] += 1
        self._propagated.add(image_id)
        return

    def propagate_feature(self, feature, node=0):
        """
        Propagates a feature, down the tree, and returns the paths in the form of node ids.
        Args:
            feature (numpy.ndarray): The feature to lookup
            root (List[int]): Node id to start the search from.
                        Default is 0, meaning the very root of the tree
        """
        path = [node]
        while self.graph.out_degree(node):  # recur, stop if leaf
            min_dist = float("inf")
            closest = None
            for child in self.graph[node]:
                distance = np.linalg.norm(
                    [self.nodes[child] - feature])  # l1 norm
                if distance < min_dist:
                    min_dist = distance
                    closest = child
            path.append(closest)
            node = closest
        return path

    def embedding(self, image):
        self.propagate(image)

        image_id = utils.get_image_id(image)

        # weights = np.array(self.graph.nodes(data="w", default=1))[:, 1]
        embedding = np.array(self.graph.nodes(data=image_id, default=0))[:, 1]

        # normalise the embeddings
        embedding = embedding / np.linalg.norm(embedding, ord=2)  # l2 norm

        return embedding  # * weights

    def subgraph(self, image_id):
        subgraph = self.graph.subgraph(
            [k for k, v in self.graph.nodes(data=image_id, default=None) if v is not None])
        colours = ["C0"] * len(self.graph.nodes)
        for node in subgraph.nodes:
            colours[node] = "C3"
        self.draw(node_color=colours)
        return subgraph

    def save(self, path=None):
        if path is None:
            path = "data"

        # store graph
        nx.write_gpickle(self.graph, os.path.join(path, "graph.pickle"))

        # store nodes with features
        with open(os.path.join(path, "nodes.pickle"), "wb") as f:
            pickle.dump(self.nodes, f)

        return True

    def draw(self, figsize=None, node_color=None, layout="tree", labels=None):
        figsize = (30, 10) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        layout = layout.lower()
        if "tree" in layout:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="dot")
        elif "radial" in layout:
            pos = nx.drawing.nx_agraph.graphviz_layout(
                self.graph, prog="twopi")
        else:
            pos = None
        if labels is None:
            nx.draw(self.graph, pos=pos, with_labels=True,
                    node_color=node_color)
        else:
            nx.draw(self.graph, pos=pos, labels=labels, node_color=node_color)
        return fig
