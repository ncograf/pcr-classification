import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from typing import List, overload
import itertools
import data_lib


@overload
def pairwise_plots(label : str, data_sets : List[str], classifier : skl.base.ClusterMixin, data_folder : str = "../Data", verbose : bool = False):
    """Plots the clustering pojected onto two dimension

    Args:
        label (str): Selected Label out of LABELS_LIST
        data_sets (List[str]): List of dataset(s) to include in the plot
        classifier (skl.base.ClusterMixin): Classifier to be used for unsupervised clustering
        data_folder (str, optional): Data folder to explore datasets. Defaults to "../Data".
        verbose (bool, optional): If true prints information about found clusters and hitting rates for labels. Defaults to False.
    """
    
    df = data_lib.load_dataset([label], data_sets, data_folder)
    
    # fist six columns are features
    df_features = df.iloc[:, :6]
    np_features = df_features.to_numpy(copy=True)
    feature_names = df_features.columns
    
    # the sixth column is interpreded as label
    df_labels = df.iloc[:, 6]
    np_labels = df_labels.to_numpy(copy=True)
    label_name = df_labels.name
    
    # predict
    preds = classifier.fit_predict(np_features)
    
    # count clusters and label outliers
    clusters_all = np.unique(preds)
    clusters = np.delete(clusters_all, clusters_all == -1)
    outliers = (preds == -1)

    # define markers for plot
    marker_map = [".", "p", "+", "v", "^", ">", "<", "1", "2", "3", "4", "8", "s", "p", "P", "*"]
    cluster_marker_index = clusters_all % len(marker_map)
    cluster_marker = [marker_map[i] for i in cluster_marker_index]
    
    # create label_match array -1 outliers, 0 true pos, 1 false pos, 2 true neg, 3 false neg
    label_match_legend = {"outlier":-1, "true pos":0, "false pos":1, "true neg":2, "false neg":3}
    label_color_map = {-1:"black", 0:"green", 1:"red", 2:"lime", 3:"magenta"}
    label_match = np.ones_like(preds) * -1 # default outlier
    
    # for each cluster associate true of false
    for cluster in clusters:
        
        # select elements in cluster
        cluster_size = sum(preds == cluster)
        cluster_num_pos = sum(np_labels[preds == cluster])
        cluster_ratio_pos = cluster_num_pos / cluster_size
        cluster_assignment = 1 if cluster_ratio_pos >= 0.5 else 0
        
        # label matches
        label_true = label_match_legend["true pos" if cluster_assignment == 1 else "true neg"]
        label_true_mask = np.logical_and(preds == cluster, np_labels == cluster_assignment)
        label_match[label_true_mask] = label_true
        label_false = label_match_legend["false pos" if cluster_assignment == 1 else "false neg"]
        label_false_mask = np.logical_and(preds == cluster, np_labels != cluster_assignment)
        label_match[label_false_mask] = label_false
        
        # gather element
        label_match_cluster = label_match[preds == cluster]
        cluster_true_pos = label_match_cluster[label_match_cluster == label_match_legend["true pos"]]
        cluster_false_pos = label_match_cluster[label_match_cluster == label_match_legend["false pos"]]
        cluster_true_neg = label_match_cluster[label_match_cluster == label_match_legend["true neg"]]
        cluster_false_neg = label_match_cluster[label_match_cluster == label_match_legend["false neg"]]
            
        if verbose:
            print(f"Cluster {cluster:>2}, num points {cluster_size:>8}, POS labelled {cluster_num_pos:>6}"
                  + f",{round(cluster_ratio_pos,5)*100:>8}%, for {label_name}, on {data_sets}.")

    # assign colors
    point_colors = np.array([label_color_map[i] for i in label_match])
    
    # poltting
    combinations = itertools.combinations(feature_names, 2)
    
    fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    # iterate over combinations for subplots
    for i, combination in enumerate(combinations): 
        df_combo_features = df_features.loc[:, combination]
        np_compo_features = df_combo_features.to_numpy()
        
        # iterate over clusters for labels
        for cluster in clusters_all:
            ax[i //3, i %3].set_xlabel(combination[0])
            ax[i //3, i %3].set_ylabel(combination[1])
            cluster_marker = marker_map[cluster % len(marker_map)]
            ax[i //3, i %3].scatter(np_compo_features[preds == cluster, 0], np_compo_features[preds == cluster, 1],\
                c = point_colors[preds == cluster], marker = cluster_marker)
    fig.tight_layout()
    plt.show()

    
def pairwise_plots(df : pd.DataFrame, label : np.array = None):
    """Plots the chosen dataframe after the transformation, pojected onto two dimension

    Args:
        label (str): Selected Label out of LABELS_LIST
        data_sets (List[str]): List of dataset(s) to include in the plot
        classifier (skl.base.ClusterMixin): Classifier to be used for unsupervised clustering
        data_folder (str, optional): Data folder to explore datasets. Defaults to "../Data".
        verbose (bool, optional): If true prints information about found clusters and hitting rates for labels. Defaults to False.
    """

    # fist six columns are features
    df_features = df.iloc[:, :6]
    feature_names = df_features.columns

    # poltting
    combinations = itertools.combinations(feature_names, 2)
    
    fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    # iterate over combinations for subplots
    for i, combination in enumerate(combinations): 
        df_combo_features = df_features.loc[:, combination]
        np_compo_features = df_combo_features.to_numpy()
        
        ax[i //3, i %3].set_xlabel(combination[0])
        ax[i //3, i %3].set_ylabel(combination[1])
        if not label is None:
            ax[i //3, i %3].scatter(np_compo_features[:,0], np_compo_features[:,1], c = label)
        else:
            ax[i //3, i %3].scatter(np_compo_features[:,0], np_compo_features[:,1])
    fig.tight_layout()
    plt.show()
    
    
def plot_cov(df : pd.DataFrame):
    """Plots covariance matrix for a Dataframe

    Args:
        df (pd.DataFrame): dataframe under consideration
    """
    f = plt.figure(figsize=(9, 5))
    plt.matshow(df.cov(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Covariance Matrix', fontsize=16)
    plt.show()