20Newsgroup documents
=====================

Introduction
------------

In this notebook, we will demonstrate use on the 20Newsgroup data. Each
document is associated with 1 of 20 newsgroup topics, organized at two
hierarchical levels.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    import pyemb as eb

Data load
---------

Import data and create dataframe.

.. code:: ipython3

    newsgroups = fetch_20newsgroups() 
    
    df = pd.DataFrame()
    df["data"] = newsgroups["data"]
    df["target"] = newsgroups["target"]
    df["target_names"] = df.target.apply(
        lambda row: newsgroups["target_names"][row])
    df[['layer1', 'layer2']] = df['target_names'].str.split('.', n=1, expand=True)

.. code:: ipython3

    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>data</th>
          <th>target</th>
          <th>target_names</th>
          <th>layer1</th>
          <th>layer2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>From: lerxst@wam.umd.edu (where's my thing)\nS...</td>
          <td>7</td>
          <td>rec.autos</td>
          <td>rec</td>
          <td>autos</td>
        </tr>
        <tr>
          <th>1</th>
          <td>From: guykuo@carson.u.washington.edu (Guy Kuo)...</td>
          <td>4</td>
          <td>comp.sys.mac.hardware</td>
          <td>comp</td>
          <td>sys.mac.hardware</td>
        </tr>
        <tr>
          <th>2</th>
          <td>From: twillis@ec.ecn.purdue.edu (Thomas E Will...</td>
          <td>4</td>
          <td>comp.sys.mac.hardware</td>
          <td>comp</td>
          <td>sys.mac.hardware</td>
        </tr>
        <tr>
          <th>3</th>
          <td>From: jgreen@amber (Joe Green)\nSubject: Re: W...</td>
          <td>1</td>
          <td>comp.graphics</td>
          <td>comp</td>
          <td>graphics</td>
        </tr>
        <tr>
          <th>4</th>
          <td>From: jcm@head-cfa.harvard.edu (Jonathan McDow...</td>
          <td>14</td>
          <td>sci.space</td>
          <td>sci</td>
          <td>space</td>
        </tr>
      </tbody>
    </table>
    </div>



For a random sample of the data, create tf-idf features.

.. code:: ipython3

    n = 5000
    df = df.sample(n=n, replace=False, random_state=22).reset_index(drop=True)

\`eb.text_matrix_and_attributes’ - creates a Y matrix of tf-idf
features. It takes in a dataframe and the column which contains the
data. Further functionality includes: removing general stopwords, adding
stopwords, removing email addresses, cleaning (lemmatize and remove
symbol, lowercase letters) and a threshold for the min/max number of
documents a word needs to appear in to be included.

.. code:: ipython3

    Y, attributes = eb.text_matrix_and_attributes(df, 'data', remove_stopwords=True, clean_text=True,
                                        remove_email_addresses=True, update_stopwords=['subject'],
                                        min_df=5, max_df=len(df)-1000)

.. code:: ipython3

    (n,p) = Y.shape
    print("n = {}, p = {}".format(n,p))


.. parsed-literal::

    n = 5000, p = 12804


Perform dimension selection using Wasserstein distances, see [1] for
details

.. code:: ipython3

    ws, dim = eb.wasserstein_dimension_select(Y, range(40), split=0.5)


.. parsed-literal::

    100%|██████████| 40/40 [02:28<00:00,  3.72s/it]

.. parsed-literal::

    Recommended dimension: 34, Wasserstein distance 0.97237


.. parsed-literal::

    


.. code:: ipython3

    print("Selected dimension: {}".format(dim))


.. parsed-literal::

    Selected dimension: 34


PCA and tSNE
------------

Now we perform PCA [1].

.. code:: ipython3

    zeta = p**-.5 * eb.embed(Y, d=dim, version='full')

Apply t-SNE

.. code:: ipython3

    from sklearn.manifold import TSNE
    
    tsne_zeta = TSNE(n_components=2, perplexity=30).fit_transform(zeta)

Make dataframes of PCA embedding and t-SNE embedding for plotting

.. code:: ipython3

    zeta_df = pd.DataFrame(zeta[:, :2])
    zeta_df["target"] = np.array(df['target_names'])
    targets = zeta_df["target"].unique()
    targets = sorted(targets)
    labels = df['target']
    
    tsne_zeta_df = pd.DataFrame(tsne_zeta)
    tsne_zeta_df["target"] = np.array(df['target_names'])
    targets = tsne_zeta_df["target"].unique()
    targets = sorted(targets)

Colours dictionary where topics from the same theme have different
shades of the same colour

.. code:: ipython3

    target_colour = {'alt.atheism': 'goldenrod',
                     'comp.graphics': 'steelblue',
                     'comp.os.ms-windows.misc': 'skyblue',
                     'comp.sys.ibm.pc.hardware': 'lightblue',
                     'comp.sys.mac.hardware': 'powderblue',
                     'comp.windows.x': 'deepskyblue',
                     'misc.forsale': 'maroon',
                     'rec.autos': 'limegreen',
                     'rec.motorcycles': 'green',
                     'rec.sport.baseball': 'yellowgreen',
                     'rec.sport.hockey': 'olivedrab',
                     'sci.crypt': 'pink',
                     'sci.electronics': 'plum',
                     'sci.med': 'orchid',
                     'sci.space': 'palevioletred',
                     'soc.religion.christian': 'darkgoldenrod',
                     'talk.politics.guns': 'coral',
                     'talk.politics.mideast': 'tomato',
                     'talk.politics.misc': 'darksalmon',
                     'talk.religion.misc': 'gold'}

Plot PCA on the LHS and PCA + t-SNE on the RHS

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for t in targets:
        t_df = zeta_df[zeta_df['target'] == t]
        ax[0].scatter(t_df[0], t_df[1], marker='o', edgecolor='black',
                      linewidth=0, s=30, label=t, c=target_colour[t])
    ax[0].set_title(f'PCA', fontsize=25)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    
    for t in targets:
        t_df = tsne_zeta_df[tsne_zeta_df['target'] == t]
        ax[1].scatter(t_df[0], t_df[1], marker='o', edgecolor='black',
                      linewidth=0, s=30, label=t, alpha=1, c=target_colour[t])
    ax[1].set_title(f'PCA + t-SNE', fontsize=25)
    ax[1].legend(loc='upper right', bbox_to_anchor=(
        1.51, 1), prop={'size': 15}, markerscale=2)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    
    plt.subplots_adjust(wspace=0.05, hspace=0)



.. image:: newsgroup_files/newsgroup_26_0.png


Hierarchical clustering with dot products [2]
---------------------------------------------

First we do this for the centroids of each topic and plot the
dendrogram. Then we do HC on the whole dataset and visualise the output
tree.

On centroids
------------

Find centroids

.. code:: ipython3

    idxs = [np.where(np.array(df['target']) == t)[0]
            for t in sorted(df['target'].unique())]
    t_zeta = np.array([np.mean(zeta[idx, :], axis=0) for idx in idxs])
    t_Y = np.array([np.mean(Y[idx, :],axis = 0) for idx in idxs]).reshape(len(sorted(df['target'].unique())),p)

Topic HC clustering

.. code:: ipython3

    t_dp_hc = eb.DotProductAgglomerativeClustering()
    t_dp_hc.fit(t_zeta);

Plot dendrogram

.. code:: ipython3

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    eb.plot_dendrogram(t_dp_hc, dot_product_clustering=True, orientation='left',
                       labels=sorted(df['target_names'].unique()))
    plt.show()



.. image:: newsgroup_files/newsgroup_35_0.png


On documents
------------

.. code:: ipython3

    dp_hc = eb.DotProductAgglomerativeClustering()
    dp_hc.fit(zeta);

Use construct tree graph from hierarchical clustering, epsilon is set to
zero as we don’t want to prune the tree

.. code:: ipython3

    tree = eb.ConstructTree(model= dp_hc, epsilon=0)
    tree.fit()


.. parsed-literal::

    Constructing tree...




.. parsed-literal::

    <pyemb.hc.ConstructTree at 0x78a385b366e0>



.. code:: ipython3

    tree.plot(labels = list(df["target_names"]), colours = target_colour, node_size=25, forceatlas_iter=0)



.. image:: newsgroup_files/newsgroup_40_0.png


References
----------


