import numpy as np
import sys
import os

class Dataset():
    def __init__(self, folder="jpg"):
        self.root = os.path.join("data", folder)
        self.filepaths = [f for f in listdir(self.root) if isfile(join(self.root, f))]
    
    def __repr__(self):
        print(self.filepaths[:5])
        
    def __getitem__(self, image_name, gray=True):
        image = cv2.imread(self.root + '/' + image_name)
        if gray:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return np.float32(gray)
        else:
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    def get_random_image(self, gray=False):
        return self.get_image_by_name(random.choice(self.filepaths),gray)

    def get_image_id(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]


class CIBR(object):
    def __init__(self, dataset, n_branches, depth):
        self.dataset = dataset
        self.n_branches = n_branches
        self.depth = depth
        self.tree = {}
        self.nodes = {}
        self.leaves = {}
        
        # private:
        self._model = KMeans(n_clusters=self.n_branches, random_state=10)
        self._current_index = 0
        
    def extract_features(self):
        # dummy features to test
        # replace with Antonio's 
        ones = np.ones(2)
        return np.array([ones, ones * 2, ones * 10, ones * 11])
        
    def create_tree(self, features=None, node=0, root=np.mean(features), current_depth=0):
        if features is None:
            features = self.extract_features()
                
        self.nodes[node] = root
        
        # if `node` is a leaf node, return
        if current_depth >= self.depth or len(features) < self.n_branches:
            return
        
        # group features by cluster
        self._model.fit(features)
        children = [[] for i in range(self.n_branches)]
        for i in range(len(features)):
            children[self._model.labels_[i]].append(features[i])
        
        # cluster children
        self.tree[node] = []
        for i in range(self.n_branches):
            self._current_index += 1
            self.tree[node].append(self._current_index)
            self.create_tree(children[i], self._current_index, self._model.cluster_centers_[i], current_depth + 1, )
        return

    def create_index(self):
        for image_path in self.dataset.image_paths:
            points, features = self.extract_features(image_path)
            for feature in features:
                leaf = self.closest_node(feature)
                image_id = self.dataset._get_image_id(image_path)
                if image_id in self.leaves[leaf]:
                    self.leaves[leaf][image_id] += 1
                else:
                    self.leaves[leaf][image_id] = 1
        return

    def closest_node(self, feature, root=0):
        """
        Returns the leaf of the tree closest to the input feature
        Args:
            feature (numpy.ndarray): The feature to lookup
            root (int): Node id to start the search from.
                        Default is 0, meaning the very root of the tree
        """
        min_dist = float("inf")
        node = None
        for child in self.tree[root]:
            distance = np.linalg.norm([self.nodes[child] - feature])
            if distance < min_dist:
                min_dist = distance
                node = child
        return node if len(self.tree[node]) == 0 else self.closest_node(feature, node)

    def encode(self, image):
        """
        Returns the representation of a new image using the vocabulary tree
        Args:
            image (numpy.ndarray): image to encode
        Return:
            (numpy.ndarray): The encoded image
        """


    def scores(self, query):
        """
        Scores the current query image against the images in the database
        Args:
            query (numpy.ndarray): Query image
        Returns:
            (numpy.ndarray): 
        """
        pass