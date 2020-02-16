class CIBR(object):
    def __init__(self, dataset, n_branches, depth):
        self.dataset = dataset
        self.n_branches = n_branches
        self.depth = depth
        self.tree = {}
        self.nodes = {}
        
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
        if current_depth >= self.depth or \
           len(features) < self.n_branches:

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

    def build_index(self):


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
        """
        pass