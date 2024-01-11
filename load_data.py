import pandas as pd
import random
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def load_data_album():
    """Loading data according to the album classification goal.
    Returns a training and a test dataset, which is split in a way so that 
    2 songs of each album are in the test set."""

    df = pd.read_csv("nightwish_lyrics.csv")
    random.seed(11)

    # build dictionary of indices
    group_indices = defaultdict(list)
    for index, row in df.iterrows():
        group_indices[row["album_id"]].append(index)

    train_indices = []
    test_indices = []


    for group, indices in group_indices.items():
        # Randomly select 2 indices for the test set
        test_indices.extend(random.sample(indices, 2))                             

        # Assign the remaining indices to the training set
        train_indices.extend([idx for idx in indices if idx not in test_indices])  

    # Create training and test datasets using the indices obtained
    training_data = df.loc[train_indices]
    test_data = df.loc[test_indices]
    return training_data, test_data

def load_data_singer():
    """Loading data according to the singer classification goal.
    Returns a training and a test dataset, which is split in a way so that 
    the test set maintains the original distribution between the classes."""

    df = pd.read_csv("nightwish_lyrics.csv")
    random.seed(11)

    # build dictionary of indices
    group_indices = defaultdict(list)
    for index, row in df.iterrows():
        group_indices[row["singer_id"]].append(index)

    train_indices = []
    test_indices = []

    for group, indices in group_indices.items():
        if(group == 1):
            # Randomly select 8 indices for the test set for Tarja, as she has most training data
            test_indices.extend(random.sample(indices, 8))                             
        else:
            # Randomly select 5 indices for the test set for other singers
            test_indices.extend(random.sample(indices, 5))    

        # Assign the remaining indices to the training set
        train_indices.extend([idx for idx in indices if idx not in test_indices])  

    # Create training and test datasets using the indices obtained
    training_data = df.loc[train_indices]
    test_data = df.loc[test_indices]
    return training_data, test_data

def load_data_era():
    """Loading data according to the era classification goal.
    Returns a training and a test dataset, which is split in a way so that 
    9 songs of each era are in the test set."""

    df = pd.read_csv("nightwish_lyrics.csv")
    random.seed(11)

    # build dictionary of indices
    group_indices = defaultdict(list)
    for index, row in df.iterrows():
        group_indices[row["era_id"]].append(index)

    train_indices = []
    test_indices = []

    for group, indices in group_indices.items():
        # Randomly select 9 indices for the test set
        test_indices.extend(random.sample(indices, 9))

        # Assign the remaining indices to the training set
        train_indices.extend([idx for idx in indices if idx not in test_indices]) 

    # Create training and test datasets using the indices obtained
    training_data = df.loc[train_indices]
    test_data = df.loc[test_indices]
    return training_data, test_data

def plot_tsne(vectors, labels, perplexity=30.0, n_iter=1000):
    """Compute and plot a t-SNE reduction of the given vectors.
    
    Arguments:
        vectors: A list of embedding vectors.
        labels: A list of class labels; must have the same length as `vectors`.
        perplexity (float): A hyperparameter of the t-SNE algorithm; recommended values
            are between 5 and 50, and can result in significantly different results.
        n_iter (int): A hyperparameter of the t-SNE algorithm, controlling the maximum
            number of iterations of the optimization algorithm.

    Returns:
        Nothing, but shows the plot.
    """

    # we have to convert the list of tensors to np arrays to be able to use them in fit_transfom of TSNE
    arrays = [tensor.numpy() for tensor in vectors]
    matrix = np.array(arrays)

    tsne = TSNE(verbose=True, perplexity=perplexity, n_iter=n_iter)
    out = tsne.fit_transform(matrix)
    
    # build a dict with all ablums or indices to display colors correctly
    label_colors = {"Angels Fall First": 'red', "Century Child": 'green', "Dark Passion Play": 'blue',
                     "Endless Forms Most Beautiful": "yellow", "Human. :||: Nature.": "grey", "Imaginaerum": "black", 
                     "Oceanborn": "brown", "Once": "orange", "Wishmaster": "pink", 1: "green", 2: "blue", 3: "red"}  
    x = out[:, 0]
    y = out[:, 1]
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(x[mask], y[mask], c=label_colors[label], label=label, alpha=0.5)
    plt.legend(loc=(1.04, 0))
