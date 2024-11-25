import os
import requests
import pickle


def _get_data_home(data_home=None):
    """
    Return the path of the dataset cache directory.
    """
    if data_home is None:
        data_home = os.environ.get('PYEMB_DATA', os.path.join(os.path.expanduser('~'), 'pyemb_data'))
    os.makedirs(data_home, exist_ok=True)
    return data_home

def _download_dataset(dataset_name):
    """
    Downloads the dataset from data github and caches it in the user's home directory.
    The file is downloaded as a whole, without using chunks.
    """
    
    github_path = 'https://raw.githubusercontent.com/pyemb/data/main/'
    url = github_path + dataset_name
    
    dataname = dataset_name.split('.')[0]
    
    # Get the path of the cache directory
    data_home = _get_data_home()

    # Define the path to save the dataset
    dataset_path = os.path.join(data_home, dataset_name)

    # Check if the dataset is already cached
    if not os.path.exists(dataset_path):
        print(f"Downloading {dataname}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an error if the request failed

            # Save the file in one go (no chunking)
            with open(dataset_path, 'wb') as f:
                f.write(response.content)  # Write the whole content at once

            print(f"{dataname} downloaded and cached in {dataset_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {dataname}: {e}")

    return dataset_path

def load_planaria():
    """
    Load the Planaria dataset. Returns a dictionary with the following keys.
    
    Returns 
    ------- 
    Y : numpy array of shape ``(n_samples, n_features)``
        The preprocessed data matrix.
    labels : numpy array of shape ``(n_samples,)``
        The cell type of each data point.
    labels : numpy array
        The unique cell types.
    colour_dict : dict
        A dictionary mapping cell types to colours.
    """
    dataset_path = _download_dataset('planaria_data.pkl')
    with open(dataset_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print('Data loaded successfully')
    return loaded_data

def load_newsgroup():
    """
    Load the Newsgroup dataset. Returns a pandas DataFrame with the following columns.
    
    Returns 
    ------- 
    data : str
        The text of the newsgroup post.
    target : int
        The label of the newsgroup post.
    target_names : str
        The label name of the newsgroup post.
    layer1: str
        The category of the newsgroup post. 
    layer2: str 
        The subcategory of the newsgroup post.
    """
    dataset_path = _download_dataset('newsgroup_data.pkl')
    with open(dataset_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print('Data loaded successfully')
    return loaded_data

def load_lyon():
    """
    Load the Lyon dataset. Returns a dictionary with the following keys.
    
    Returns
    ------- 
    data : numpy array of shape ``(n_edges, 3)``
        The edges of the network. The first column is time and the second and third columns are the nodes. The nodes are indices from 0.
    labels : numpy array of shape ``(n_nodes,)`` 
        The labels of the nodes. The index of the label corresponds to the node index.
    """
    dataset_path = _download_dataset('lyon_data.pkl')
    with open(dataset_path, 'rb') as file:
        loaded_data = pickle.load(file)
        print('Data loaded successfully')
    return loaded_data