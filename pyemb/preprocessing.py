import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from copy import deepcopy
from tqdm import tqdm

from ._utils import *

def graph_from_dataframes(
    tables,
    relationship_cols,
    same_attribute=False,
    dynamic_col=None,
    weight_col=None,
    join_token="::",
):
    """
    Create a graph from a list of tables and relationships.

    Parameters
    ----------
    tables : list of pandas.DataFrame
        The list of tables.
    relationship_cols : list of lists
        The list of relationships. Either: Each relationship is a list of two lists,
        each of which contains the names of the columns in the corresponding table. Or, a list of lists and each pair is looked for in each table.
    same_attribute : bool
        Whether the entities in the columns are from the same attribute.
    dynamic_col : list of str
        The list of dynamic columns.
    weight_col : list of str
        The list of weight columns.
    join_token : str
        The token used to join the names of the partitions and the names of the nodes.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in the rows. The second list contains
        the attributes of the nodes in the columns.
    """
    # Ensure data and relationship_cols are in list format
    # add valueerror if not in the correct format
    if not isinstance(tables, list):
        tables = [tables]
    if isinstance(relationship_cols[0], str):
        relationship_cols = [relationship_cols]
    if not isinstance(relationship_cols[0][0], list):
        relationship_cols = [relationship_cols] * len(tables)
    # Handle the case when dynamic_col is None
    if dynamic_col is None:
        dynamic_col = [None] * len(tables)
    elif isinstance(dynamic_col, str):
        dynamic_col = [dynamic_col] * len(tables)
    if len(dynamic_col) != len(tables):
        dynamic_col = dynamic_col * len(tables)
    # Handle the case when weight_col is None
    if weight_col is None:
        weight_col = [None] * len(tables)
    elif isinstance(weight_col, str):
        weight_col = [weight_col] * len(tables)
    if len(weight_col) != len(tables):
        weight_col = weight_col * len(tables)

    edge_list = _create_edge_list(
        tables, relationship_cols, same_attribute, dynamic_col, join_token, weight_col
    )
    nodes, partitions, times, node_ids, time_ids = _extract_node_time_info(
        edge_list, join_token
    )

    edge_list = _transform_edge_data(edge_list, node_ids, time_ids, len(nodes))
    A = _create_adjacency_matrix(edge_list, len(nodes), len(times), weight_col)
    attributes = _create_node_attributes(
        nodes, partitions, times, len(nodes), len(times)
    )

    return _unfolded_to_list(A.tocsr()), attributes



def _create_edge_list(
    tables, relationship_cols, same_attribute, dynamic_col, join_token, weight_col
):
    """
    Create an edge list from a list of tables and relationships.

    Parameters
    ----------
    tables : list of pandas.DataFrame
        The list of tables.
    relationship_cols : list of lists
        The list of relationships. Each relationship is a list of two lists,
        each of which contains the names of the columns in the corresponding table.
    dynamic_col : list of str
        The list of dynamic columns.
    join_token : str
        The token used to join the names of the partitions and the names of the nodes.
    weight_col : list of str
        The list of weight columns.

    Returns
    -------
    edge_list : pandas.DataFrame
        The edge list.
    """

    edge_list = []
    for data0, relationship_cols0, dynamic_col0, weight_col0 in tqdm(
        zip(tables, relationship_cols, dynamic_col, weight_col)
    ):
        for partition_pair in relationship_cols0:
            if set(partition_pair).issubset(data0.columns):

                cols = deepcopy(partition_pair)
                colnames = ["V1", "V2"]

                if dynamic_col0:
                    cols.append(dynamic_col0)
                if weight_col0:
                    cols.append(weight_col0)
                    colnames.append("W")

                pair_data = deepcopy(data0[cols].drop_duplicates())
                if not dynamic_col0:
                    pair_data["T"] = np.nan
                colnames.append("T")

                pair_data.columns = colnames

                if not same_attribute:
                    p1 = partition_pair[0]
                    p2 = partition_pair[1]
                else:
                    p1 = p2 = partition_pair[0]
                pair_data["V1"] = [f"{p1}{join_token}{x}" for x in pair_data["V1"]]
                pair_data["V2"] = [f"{p2}{join_token}{x}" for x in pair_data["V2"]]
                pair_data["P1"] = partition_pair[0]
                pair_data["P2"] = partition_pair[1]

                edge_list.append(pair_data)
                # print(partition_pair)
    return pd.concat(edge_list)


def find_subgraph(A, attributes, subgraph_attributes):
    """
    Find a subgraph of a multipartite graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the multipartite graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    subgraph_attributes : list of lists
        The attributes of the nodes of the wanted in the subgraph. The first list contains
        the attributes of the nodes wanted in the rows. The second
        list contains the attributes of the nodes wanted in the column.

    Returns
    -------
    subgraph_A : scipy.sparse.csr_matrix
        The adjacency matrix of the subgraph.
    subgraph_attributes : list of lists
        The attributes of the nodes of the subgraph. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """

    if not isinstance(subgraph_attributes[0], list):
        subgraph_attributes[0] = [subgraph_attributes[0]]

    if not isinstance(subgraph_attributes[1], list):
        subgraph_attributes[1] = [subgraph_attributes[1]]

    # find the indices of the rows with required attributes
    subgraph_node_indices_row = []
    for node_idx, node_attributes in enumerate(attributes[0]):
        for each_subgraph_attributes in subgraph_attributes[0]:
            matched = True
            for key, value in each_subgraph_attributes.items():
                if key not in node_attributes or node_attributes[key] != value:
                    matched = False
                    break
            if matched:
                subgraph_node_indices_row.append(node_idx)

    # find the indices of the columns with required attributes
    subgraph_node_indices_col = []
    for node_idx, node_attributes in enumerate(attributes[1]):
        for each_subgraph_attributes in subgraph_attributes[1]:
            matched = True
            for key, value in each_subgraph_attributes.items():
                if key not in node_attributes or node_attributes[key] != value:
                    matched = False
                    break
            if matched:
                subgraph_node_indices_col.append(node_idx)

    subgraph_A, subgraph_attributes = _subgraph_idx(
        A, attributes, subgraph_node_indices_row, subgraph_node_indices_col
    )

    return subgraph_A, subgraph_attributes


def _subgraph_idx(A, attributes, idx0, idx1):
    """
    Find a subgraph of a multipartite graph by indices.
    """
    subgraph_A = A[np.ix_(idx0, idx1)]
    subgraph_attributes = [
        [attributes[0][i] for i in idx0],
        [attributes[1][i] for i in idx1],
    ]
    return subgraph_A, subgraph_attributes


def find_connected_components(A, attributes, n_components=None):
    """
    Find connected components of a multipartite graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    n_components : int
        The number of components to be found.

    Returns
    -------
    cc_As : list of scipy.sparse.csr_matrix
        The adjacency matrices of the connected components.
    cc_attributes : list of lists
        The attributes of the nodes of the connected components. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """

    A_dilation = _symmetric_dilation(A)
    _, cc = sparse.csgraph.connected_components(A_dilation)
    print(f"Number of connected components: {_}")
    cc = [cc[: A.shape[0]], cc[A.shape[0] :]]
    if n_components == None:
        n_components = _
    cc_As = []
    cc_attributes = []
    if n_components == 1:
        cc_As = A
        cc_attributes = attributes
    else:
        for i in range(n_components):
            idx0 = np.where(cc[0] == i)[0]
            idx1 = np.where(cc[1] == i)[0]
            store_cc_A, store_cc_attributes = _subgraph_idx(A, attributes, idx0, idx1)
            cc_As.append(store_cc_A)
            cc_attributes.append(store_cc_attributes)

    return cc_As, cc_attributes


def largest_cc_of(A, attributes, partition, dynamic=False):
    """
    Find the connected component containing the most nodes from a partition.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    partition : str
        The partition to be searched.
    dynamic : bool
        Whether we want the connected component containing the most nodes from dynamic part or not.

    Returns
    -------
    cc_A : scipy.sparse.csr_matrix
        The adjacency matrix of the connected component.
    cc_attributes : list of lists
        The attributes of the nodes of the connected component. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """
    cc_As, cc_attributes = find_connected_components(A, attributes)
    if not dynamic:
        attrs = [att[0] for att in cc_attributes]
    else:
        attrs = [att[1] for att in cc_attributes]

    counts = [_count_based_on_keys(att, "partition") for att in attrs]
    select_idx = np.argmax([c.get(partition, 0) for c in counts])
    return cc_As[select_idx], cc_attributes[select_idx]


def to_networkx(A, attributes, symmetric=None):
    """
    Convert a multipartite graph to a networkx graph.
    """
    if symmetric is None:
        symmetric = (A != A.T).nnz == 0

    if symmetric:
        G_nx = nx.Graph(A)
        nx.set_node_attributes(G_nx, {i: a for i, a in enumerate(attributes[0])})
    else:
        n0 = len(attributes[0])
        n1 = len(attributes[1])
        G_nx = nx.Graph(_symmetric_dilation(A))
        nx.set_node_attributes(G_nx, {i: a for i, a in enumerate(attributes[0])})
        nx.set_node_attributes(G_nx, {i + n0: a for i, a in enumerate(attributes[1])})
        nx.set_node_attributes(G_nx, {i: {"bipartite": 0} for i in range(n0)})
        nx.set_node_attributes(G_nx, {i + n0: {"bipartite": 1} for i in range(n1)})
    return G_nx


def time_series_matrix_and_attributes(data, time_col, drop_nas=True):
    """
    Create a matrix from a time series.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be used to create the matrix.
    time_col : str
        The name of the column containing the time information.
    drop_nas : bool
        Whether to drop rows with missing values.

    Returns
    -------
    Y : numpy.ndarray
        The matrix created from the time series.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    """
    data = data.sort_values(by=time_col)
    if drop_nas:
        data = data.dropna(axis=1, how="any")

    times = list(data[time_col])
    data.drop([time_col], axis=1, inplace=True)
    ids = list(data.columns)

    Y = np.array(data).T
    attributes = [[{"name": i} for i in ids], [{"time": i} for i in times]]
    return Y, attributes

def text_matrix_and_attributes(
    data,
    column_name,
    remove_stopwords=True,
    clean_text=True,
    remove_email_addresses=False,
    update_stopwords=None,
    **kwargs,
):
    """
    Create a matrix from a column of text data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be used to create the matrix.
    column_name : str
        The name of the column containing the text data.
    remove_stopwords : bool
        Whether to remove stopwords.
    clean_text : bool
        Whether to clean the text data.
    remove_email_addresses : bool
        Whether to remove email addresses.
    update_stopwords : list of str
        The list of additional stopwords to be removed.
    kwargs : dict
        Other arguments to be passed to sklearn.feature_extraction.text.TfidfVectorizer.

    Returns
    -------
    Y : numpy.ndarray
        The matrix created from the text data.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    """

    _ensure_stopwords_downloaded()

    # gets rid of email addresses  in data
    if remove_email_addresses:
        data[column_name] = data[column_name].apply(_del_email_address)

    if clean_text:
        # gets rid of stopwords, symbols, makes lower case and base words
        data[column_name] = data[column_name].apply(_clean_text_)

    if remove_stopwords:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        if update_stopwords:
            stopwords.update(update_stopwords)
    else:
        stopwords = None

    # make matrix from text using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=list(stopwords), **kwargs)
    Y = vectorizer.fit_transform(data[column_name])

    # attributes
    row_attrs = [{"document": i} for i in data.index]
    col_attrs = [{"term": i} for i in vectorizer.get_feature_names_out()]

    return Y, [row_attrs, col_attrs]

