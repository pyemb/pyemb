Planaria single-cell
====================

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import networkx as nx
    
    import pyemb as eb

.. code:: ipython3

    Y = np.array(pd.read_csv('data/planaria_sample_data.csv', index_col=0))
    (n,p) = Y.shape
    print('Data matrix is', n, 'samples by', p, 'features')
    
    labels = np.genfromtxt('data/sample_annotation_5000.txt', delimiter=',', dtype=str) 
    ordered_cats = np.genfromtxt('data/planaria_ordered_cats.csv', delimiter=',', dtype=str)
    
    colors = pd.read_csv('data/colors_dataset.txt', header=None, sep='\t')
    colors = {k: c for k, c in colors.values}


.. parsed-literal::

    Data matrix is 5000 samples by 5821 features


.. code:: ipython3

    dim = 20
    zeta = p**-.5 * eb.embed(Y, d=dim, version='full')

.. code:: ipython3

    tree = eb.ConstructTree(zeta, epsilon=0.25)
    tree.fit()


.. parsed-literal::

    Performing clustering...
    Calculating branch lengths...
    Constructing tree...




.. parsed-literal::

    <pyemb.hc.ConstructTree at 0x7e4cb35c81c0>



.. code:: ipython3

    tree.plot(labels,colors, prog = 'twopi')


.. parsed-literal::

    100%|██████████| 250/250 [00:03<00:00, 79.07it/s]


.. parsed-literal::

    BarnesHut Approximation  took  1.92  seconds
    Repulsion forces  took  1.03  seconds
    Gravitational forces  took  0.02  seconds
    Attraction forces  took  0.01  seconds
    AdjustSpeedAndApplyForces step  took  0.09  seconds



.. image:: planaria_files/planaria_5_2.png

