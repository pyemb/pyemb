import numpy as np
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import networkx as nx
from scipy.stats import kendalltau
import matplotlib.pyplot as plt


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
    
## ======= linkage matrix and Kendall's tau similarity ======= ##

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




# def plot_HC_clustering(model, node_colours=None, no_merges=None, labels=None, plot_labels=None, internal_node_colour='black',
#                        linewidths=None, edgecolors=None, leaf_node_size=20, fontsize=10, internal_node_size=1, figsize=(10, 10)):
#     """ 
#     Plot the hierarchical clustering tree.    

#     Parameters  
#     ----------  
#     model : AgglomerativeClustering 
#         The fitted model.   
#     node_colours : array-like, shape (n_samples,)   
#         The colour of the nodes.
#     no_merges : int, optional
#         The number of merges to plot. If None, plot all merges.
#     labels : array-like, shape (n_samples,) 
#         The labels of the samples.
#     plot_labels : bool, optional    
#         Whether to plot the labels.
#     internal_node_colour : str, optional    
#         The colour of the internal nodes.   
#     linewidths : float, optional    
#         The width of the lines. 
#     edgecolors : str, optional  
#         The colour of the edges.
#     leaf_node_size : int, optional  
#         The size of the leaf nodes. 
#     fontsize : int, optional
#         The size of the font.
#     internal_node_size : int, optional  
#         The size of the internal nodes.
#     figsize : tuple, optional   
#         The size of the figure.

#     Returns 
#     ------- 
#     None    
#     """

#     if no_merges is None:
#         no_merges = model.children_.shape[0]
#     if node_colours is None:
#         node_colours = np.repeat('skyblue', model.n_leaves_)
#     if labels is None:
#         labels = np.repeat('', model.children_.shape[0] + model.n_leaves_)

#     data = model.children_[:no_merges, :]
#     n = model.n_leaves_

#     G = nx.Graph()

#     for i in range(data.shape[0]):
#         idx = i + n
#         to_merge = data[i]
#         G.add_edge(to_merge[0], idx)
#         G.add_edge(to_merge[1], idx)

#     node_colours_ = {}
#     node_sizes = {}
#     node_names = {}
#     for node in G.nodes():
#         if node < n:
#             node_colours_[node] = node_colours[node]
#             node_sizes[node] = leaf_node_size
#             node_names[node] = labels[node]
#         else:
#             node_colours_[node] = internal_node_colour
#             node_sizes[node] = internal_node_size
#             node_names[node] = ''

#     # Draw the graph with node sizes and names
#     plt.figure(figsize=figsize)  # Adjust figure size

#     positions = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
#     nx.draw(G, positions, with_labels=False, node_size=[
#             node_sizes[node] for node in G.nodes()], node_color=[
#             node_colours_[node] for node in G.nodes()], edgecolors=edgecolors, linewidths=linewidths)

#     nx.draw_networkx_labels(
#         G, positions, labels=node_names, font_size=fontsize)
#     plt.show()



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



def _get_triu(matrix, k=1):
    return matrix[np.triu_indices(matrix.shape[0], k=k)]


def _utri2mat(utri):
    n = int(-1 + np.sqrt(1 + 8*len(utri))) // 2
    iu1 = np.triu_indices(n)
    ret = np.empty((n, n))
    ret[iu1] = utri
    ret.T[iu1] = utri
    return ret