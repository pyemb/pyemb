import numpy as np
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from fa2_modified import ForceAtlas2

from ._utils import _is_visited, _set_visited, _find_colours, _find_cluster_sizes


## ========= Dot product based hierarchical clustering ========= ##

class DotProductAgglomerativeClustering:
    """ 
    Perform hierarchical clustering using dot product as the metric.    
    
    Parameters: 
    ----------  
    metric : str, optional  
        The metric to use for clustering.   
    linkage : str, optional 
        The linkage criterion to use.
    distance_threshold : float, optional    
        The linkage distance threshold above which, clusters will not be merged.
    n_clusters : int, optional  
        The number of clusters to find.
        
    Attributes: 
    ----------  
    distances_ : ndarray    
        The distances between the clusters. 
    children_ : ndarray 
        The children of each non-leaf node.
    labels_ : ndarray   
        The labels of each point.
    n_clusters_ : int   
        The number of clusters.
    n_connected_components_ : int   
        The number of connected components. 
    n_leaves_ : int 
        The number of leaves.
    n_features_in_ : int    
        The number of features seen during fit.
    n_clusters_ : int   
        The number of clusters.
    """
    def __init__(self, metric='dot_product', linkage='average', distance_threshold=0, n_clusters=None):
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.n_clusters = n_clusters
        self.model = None
        

    def fit(self, X):
        model = AgglomerativeClustering(
            metric=self._ip_metric if self.metric == 'dot_product' else self.metric,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            n_clusters=self.n_clusters
        )
        model.fit(X)
        
        # If distances_ attribute is available, adjust it as specified
        if hasattr(model, 'distances_'):
            self.distances_ = -model.distances_
        if hasattr(model, 'distances_'):
            self.children_ = model.children_
        if hasattr(model, 'labels_'):
            self.labels_ = model.labels_    
        if hasattr(model, 'n_clusters_'):
            self.n_clusters_ = model.n_clusters_
        if hasattr(model, 'n_connected_components_'):   
            self.n_connected_components_ = model.n_connected_components_    
        if hasattr(model, 'n_leaves_'): 
            self.n_leaves_ = model.n_leaves_    
        if hasattr(model, 'n_features_in_'):    
            self.n_features_in_ = model.n_features_in_  
        if hasattr(model, 'n_clusters_'):   
            self.n_clusters_ = model.n_clusters_    
        
        self = model
        return self

    @staticmethod
    def _ip_metric(X):
        return -(X @ X.T)
    
## ======= Linkage matrix ======= ##


def linkage_matrix(model):
    """ 
    Convert a hierarchical clustering model to a linkage matrix.    
    
    Parameters: 
    ----------  
    model : AgglomerativeClustering
        The fitted model.   
    get_heights : bool, optional    
        Whether to return heights or counts.
    max_height : float, optional    
        The maximum height of the tree.
        
    Returns:    
    ------- 
    ndarray
        The linkage matrix.
    """ 
    counts = np.zeros(model.children_.shape[0])
    n_samples = model.children_.shape[0] + 1
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    linkage_matrix[:,:2] = linkage_matrix[:,:2].astype(int)

    return linkage_matrix

## ======= Cophenetic distances ======= ##

def cophenetic_distances(Z):
    """
    Calculate the cophenetic distances between each observation and internal nodes.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.

    Returns
    -------
    d : ndarray
        The full distance matrix (2n-1) x (2n-1).
    """
    n = Z.shape[0] + 1
    N = 2 * n - 1
    d = np.zeros((N, N))
    curr_node = np.zeros(n, dtype=int)
    members = np.zeros(n, dtype=int)
    left_start = np.zeros(n, dtype=int)

    visited_size = ((N >> 3) + 1)
    visited = np.zeros(visited_size, dtype=np.uint8)

    k = 0
    curr_node[0] = 2 * n - 2
    left_start[0] = 0
    while k >= 0:
        root = curr_node[k] - n
        i_lc = int(Z[root, 0])
        i_rc = int(Z[root, 1])

        if i_lc >= n:  # left child is not a leaf
            n_lc = int(Z[i_lc - n, 3])
            if not _is_visited(visited, i_lc):
                _set_visited(visited, i_lc)
                k += 1
                curr_node[k] = i_lc
                left_start[k] = left_start[k - 1]
                continue  # visit the left subtree
        else:
            n_lc = 1
            members[left_start[k]] = i_lc

        if i_rc >= n:  # right child is not a leaf
            n_rc = int(Z[i_rc - n, 3])
            if not _is_visited(visited, i_rc):
                _set_visited(visited, i_rc)
                k += 1
                curr_node[k] = i_rc
                left_start[k] = left_start[k - 1] + n_lc
                continue  # visit the right subtree
        else:
            n_rc = 1
            members[left_start[k] + n_lc] = i_rc

        # back to the root of current subtree
        dist = Z[root, 2]
        right_start = left_start[k] + n_lc

        # Update distances for leaf nodes in left and right children
        for i in range(left_start[k], right_start):
            for j in range(right_start, right_start + n_rc):
                d[members[i], members[j]] = dist
                d[members[j], members[i]] = dist

        # Update distances for internal nodes
        all_members = members[left_start[k]:right_start + n_rc]
        for i in all_members:
            d[root + n, i] = dist
            d[i, root + n] = dist

        # Add distance between the root of the current subtree and its parent
        if k > 0:
            parent = curr_node[k - 1] - n
            parent_dist = Z[parent, 2]
            d[root + n, parent + n] = parent_dist
            d[parent + n, root + n] = parent_dist

        k -= 1  # back to parent node

    return d


## ======= Calculate branch lengths ======= ##

def branch_lengths(Z, point_cloud = None):
    """
    Calculate branch lengths for a hierarchical clustering dendrogram.

    Parameters:
    ----------
    Z (ndarray): The linkage matrix.
    point_cloud (ndarray): The data points.

    Returns:
    -------
    ndarray: Matrix of branch lengths.
    """
    n = Z.shape[0] + 1
    N = 2 * n - 1
    
    
    if point_cloud is None:
        leaf_heights = np.repeat(np.max(Z[:, 2]), n)
    else:
        leaf_heights = np.linalg.norm(point_cloud, axis=1)**2
    heights = np.hstack([leaf_heights, Z[:, 2]])
    merge_heights = cophenetic_distances(Z)

    # Efficient matrix calculation
    heights_matrix = np.add.outer(heights, heights)
    B = np.abs(heights_matrix - 2 * merge_heights)
    B[merge_heights == 0] = -np.inf

    return B

## ======= Find node descendants  ======= ##

def find_descendents(Z, node, desc=None, just_leaves=True):
    """
    Find all descendants of a given node in a hierarchical clustering tree.

    Parameters:
    ----------
    Z (ndarray): The linkage matrix.
    node (int): The node to find descendants of.
    desc (dict, optional): Dictionary to store descendants.
    just_leaves (bool, optional): Whether to include only leaf nodes.

    Returns:
    -------
    list: List of descendants.
    """
    if desc is None:
        desc = {}
    n_samples = Z.shape[0] + 1
    if node in desc:
        return desc[node]
    if node < n_samples:
        return [node]
    
    pair = Z[node - n_samples, :2].astype(int)
    desc[node] = find_descendents(Z, pair[0], desc, just_leaves) + find_descendents(Z, pair[1], desc, just_leaves)

    if not just_leaves:
        desc[node] = [int(pair[0]), int(pair[1])] + desc[node]
    return desc[node]

## ======= Construct epsilon tree ======= ##

def _epsilon_tree(Z, B, epsilon = 0.25):
    """
    Condense a hierarchical clustering tree.

    Parameters:
    ----------
    Z (ndarray): The linkage matrix.
    B (ndarray): Matrix of branch lengths.
    epsilon (float): Threshold for condensing the tree.

    Returns:
    -------
    nx.Graph: Condensed tree as a NetworkX graph.
    """
    n = Z.shape[0] + 1
    N = 2*n-1
    G = nx.Graph()

    reverse_lm = Z[::-1] 

    internal_nodes = sorted(range(n, n + reverse_lm.shape[0]), reverse=True)
    internal_nodes_set = set(internal_nodes)

    desc = {i: find_descendents(Z, i) for i in range(n, 2 * n - 1)}
    desc[N-1] = list(range(N))
    
    while len(internal_nodes) > 0:
        idx = internal_nodes[0]        
        i = N - idx - 1
        merge = reverse_lm[i]
        left = int(merge[0])
        right = int(merge[1])
        
        left_desc = desc[left] if left >= n else [left]
        right_desc = desc[right] if right >= n else [right]
        
        if np.any(B[idx, left_desc] > epsilon):
            G.add_edge(idx, left, len=B[idx, left])
        else:
            internal_nodes_set.difference_update([left] + left_desc)
        
        if np.any(B[idx, right_desc] > epsilon):
            G.add_edge(idx, right, len=B[idx, right])
        else:
            internal_nodes_set.difference_update([right] + right_desc)
        
        # Update the list after potential changes
        internal_nodes_set.difference_update([idx])
        internal_nodes = sorted(internal_nodes_set, reverse=True)
    return G

## ======= Find clusters ======= ##

def _find_clusters(G, Z, just_leaves=True):
    """
    Find clusters in a condensed tree.

    Parameters:
    G (nx.Graph): Condensed tree.
    Z (ndarray): The linkage matrix.
    just_leaves (bool, optional): Whether to include only leaf nodes.

    Returns:
    dict: Dictionary of clusters.
    """
    total = []
    visited = set()
    G_desc = {}

    for left, right in reversed(list(G.edges(data=False))):
        for node in (right, left):
            if node not in visited:
                desc = find_descendents(Z, node, just_leaves=just_leaves)
                G_desc[node] = list(set(desc) - set(total))
                total += desc
                visited.add(node)
    return G_desc

## ======= Construct tree ======= ##

class ConstructTree:
    """
    Construct a condensed tree from a hierarchical clustering model.
    
    Parameters: 
    ----------  
    model : AgglomerativeClustering, optional  
        The fitted model.   
    point_cloud : ndarray, optional 
        The data points.
    epsilon : float, optional   
        The threshold for condensing the tree.
    **kwargs : dict, optional   
        Additional keyword arguments.
    
    Attributes: 
    ----------  
    model : AgglomerativeClustering  
        The fitted model.   
    point_cloud : ndarray   
        The data points.
    epsilon : float 
        The threshold for condensing the tree.
    linkage : ndarray   
        The linkage matrix. 
    tree : nx.Graph 
        The condensed tree.
    collapsed_branches : dict   
        The collapsed branches.    
    """
    def __init__(self, point_cloud = None,  model = None, epsilon=0.25):
        self.model = model
        self.point_cloud = point_cloud
        self.epsilon = epsilon
        self.linkage = None
        self.tree = None
        self.collapsed_branches = None

    def fit(self, **kwargs):
        """
        Fit the condensed tree.
        """
        if self.model is None and self.point_cloud is None:
            raise ValueError("Please provide either an agglomerative clustering or the data for hierchical clustering.")
        
        if self.model is not None and self.point_cloud is not None:
            Z = linkage_matrix(self.model)
            print('Calculating branch lengths...')
            B = branch_lengths(Z, self.point_cloud)
            print('Constructing tree...')
            self.tree = _epsilon_tree(Z, B, epsilon = self.epsilon)
        if self.model is None and self.point_cloud is not None:
            print('Performing clustering...')
            self.model = DotProductAgglomerativeClustering(**kwargs)
            self.model.fit(self.point_cloud)
            Z = linkage_matrix(self.model)
            print('Calculating branch lengths...')
            B = branch_lengths(Z, self.point_cloud)
            print('Constructing tree...')
            self.tree = _epsilon_tree(Z, B, epsilon = self.epsilon)
        if self.model is not None and self.point_cloud is None:
            print('Constructing tree...')
            data = self.model.children_
            n = self.model.n_leaves_
            self.tree = nx.Graph()
            for i in range(self.point_cloud.shape[0]):
                idx = i + n
                to_merge = self.point_cloud[i]
                self.tree.add_edge(to_merge[0], idx)
                self.tree.add_edge(to_merge[1], idx)
        return self
    
    def plot(self, labels = None, colours = None, colour_threshold = .5, prog = "sfdp", forceatlas_iter = 250, node_size = 10, scaling_node_size = 1, **kwargs):
        """
        Plot the condensed tree.
        """
        if self.tree is None:
            raise ValueError("Please fit the tree first.")
        if self.linkage is None:
            self.linkage = linkage_matrix(self.model)
        
        G_clusters = _find_clusters(self.tree, self.linkage, just_leaves = True)
        
        if labels is not None and isinstance(colours, dict):
            colour_dict = _find_colours(labels, colours, G_clusters, colour_threshold = colour_threshold)
            colours = [colour_dict[node] for node in self.tree.nodes()]
        if colours is None:
            colours = ['lightblue' if node < self.model.n_leaves_  else 'black' for node in self.tree.nodes()]
        
        cluster_sizes_dict = _find_cluster_sizes(G_clusters)
        plot_sizes_dict = {k: node_size + scaling_node_size * v for k, v in cluster_sizes_dict.items()}
        sizes = [plot_sizes_dict[node] for node in self.tree.nodes()]
    
        n = self.model.n_leaves_
        forceatlas2 = ForceAtlas2()
        plt.figure(figsize=(10,10))
        positions = nx.nx_agraph.graphviz_layout(self.tree, prog=prog, root=2*n-2)
        if forceatlas_iter != 0:
            positions = forceatlas2.forceatlas2_networkx_layout(self.tree, pos=positions, iterations=forceatlas_iter)
        nx.draw(self.tree, positions, node_color=colours, 
                node_size=sizes, **kwargs)
        plt.show()





## ======= Kendall's tau similarity ======= ##
    
def get_ranking(model):
    """ 
    Get the ranking of the samples. 
    
    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.
        
    Returns 
    ------- 
    mh_rank : numpy.ndarray 
        The ranking of the samples.
    """
    lm = linkage_matrix(model)
    cophenetic_dists = squareform(cophenet(lm))
    mh_rank = np.array([rankdata(cophenetic_dists[i], method='dense')
                       for i in range(cophenetic_dists.shape[0])])
    mh_rank = mh_rank - 1
    return mh_rank


def kendalltau_similarity(model, true_ranking):
    """ 
    Calculate the Kendall's tau similarity between the model and true ranking.  

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.   
    true_ranking : array-like, shape (n_samples, n_samples) 
        The true ranking of the samples.

    Returns 
    ------- 
    float   
        The mean Kendall's tau similarity between the model and true ranking.
    """

    if model.labels_.shape[0] != true_ranking.shape[0]:
        raise ValueError(
            "The number of samples in the model and true_ranking must be the same.")
    n = model.labels_.shape[0]

    ranking = get_ranking(model)
    kt = [kendalltau(ranking[i], true_ranking[i]
                     ).correlation for i in tqdm(range(ranking.shape[0]))]
    return np.mean(kt)


## ========= Hyperbolicity ========= ##


def sample_hyperbolicity(data, metric='dot_products', num_samples=5000):
    """ 
    Calculate the hyperbolicity of the data.    
    
    Parameters  
    ----------  
    data : numpy.ndarray    
        The data to calculate the hyperbolicity.
    metric : str    
        The metric to use. Options are 'dot_products', 'cosine_similarity', 'precomputed' or any metric supported by scikit-learn.   
    num_samples : int   
        The number of samples to calculate the hyperbolicity.   
        
    Returns 
    ------- 
    float   
        The hyperbolicity of the data.
    """
    

    valid_distances = ['dot_products', 'cosine_similarity', 'precomputed', 'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard',
                       'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine', 'kulsinski']

    if metric not in valid_distances:
        raise ValueError(
            'Invalid distance metric, please choose from {}'.format(valid_distances))

    if metric == 'dot_products':
        initial_mhs = data @ data.T
    if metric == 'cosine_similarity':
        initial_mhs = cosine_similarity(data)
    if metric == 'precomputed':
        initial_mhs = data
    if metric != 'dot_products' and metric != 'cosine_similarity' and metric != 'precomputed':
        initial_mhs = - pairwise_distances(data, metric=metric)

    n = data.shape[0]

    print('Calculating distance matrix')
    ranks = rankdata(_get_triu(initial_mhs, k=0), method='average')
    c_ranks = ranks / np.max(ranks)
    mhs = _utri2mat(c_ranks)
    heights = np.repeat(np.diag(mhs), n).reshape((n, n))
    distance_matrix = heights + heights.T - 2*mhs
    print(np.min(distance_matrix), np.max(distance_matrix))

    hyps = []
    print('Calclating hyperbolicity')
    for i in tqdm(range(num_samples)):
        node_tuple = np.random.choice(range(n), 4, replace=False)
        try:
            d01 = distance_matrix[node_tuple[0], node_tuple[1]]
            d23 = distance_matrix[node_tuple[2], node_tuple[3]]
            d02 = distance_matrix[node_tuple[0], node_tuple[2]]
            d13 = distance_matrix[node_tuple[1], node_tuple[3]]
            d03 = distance_matrix[node_tuple[0], node_tuple[3]]
            d12 = distance_matrix[node_tuple[1], node_tuple[2]]

            s = [d01 + d23, d02 + d13, d03 + d12]
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    # return hyps
    return np.max(hyps), np.mean(hyps)

