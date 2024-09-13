Installation and Overview
==========================


Installation    
------------

To install the package, you can use pip

.. code:: ipython3

    pip install pyemb

To install the package from source, clone the repository and run

.. code:: ipython3
    
        git clone   
        cd pyemb  
        pip install -e .    

Overview    
--------    

The package includes the following modules: 
    * preprocessing 
    * matrix and graph tools 
    * embedding 
    * visualisation
    * hierarchical clustering    

The functionality of the package is demonstrated in the tutorials through a few real datasets.

Preprocessing
~~~~~~~~~~~~~~

This module contains a variety of functions for preprocessing data with two outputs
    (1) **matrix* of the data**,
    (2) **list of two dictionaries** , one for the rows and one for the columns, that contain the metadata of the data.

The types of data that can be processed include:
    * **relational database**: `graph_from_dataframes`, where pairs of columns are specified to indicate nodes in the same row have an edge between them,
    * **time series**: `time_series_matrix_and_attributes` *(in progress)*,
    * **text data**: `text_matrix_and_attributes`` where the column on text data is converted to tf-idf features (columns).

There is also functionality for **finding connected components, subgraphs and converting to a networkx object**. 

Matrix and Graph Tools 
~~~~~~~~~~~~~~~~~~~~~~~



Embedding
~~~~~~~~~~

Visualisation   
~~~~~~~~~~~~~~

Hierarchical Clustering 
~~~~~~~~~~~~~~~~~~~~~~~ 

Simulation
~~~~~~~~~~