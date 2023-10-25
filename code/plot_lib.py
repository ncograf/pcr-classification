import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def pairwise_plots_pred_true(df : pd.DataFrame, predicted_label : npt.ArrayLike, true_label : npt.ArrayLike):
    """Plots the chosen dataframe after the transformation, pojected onto two dimension

    True positives: green
    False positivs: red
    True negatives: back
    False negatives: purple

    Args:
        df (pd.DataFrame): points to be plotted
        predicted_label (npt.ArrayLike): labels predicted by some algorithm
        true_label (npt.ArrayLike): labels according to "ground truth"
    """

    predicted_label = np.array(predicted_label)
    true_label = np.array(true_label)

    assert true_label.shape == predicted_label.shape

    # fist six columns are features
    df_features = df.iloc[:, :6]
    feature_names = df_features.columns
    
    # false positives = 1, false negatives = -1
    false_labels = predicted_label - true_label
    
    # create colors
    color_labels = pd.Series(true_label, copy=True, dtype=str)
    color_labels.iloc[true_label == 1] = '#15B01A' # green
    color_labels.iloc[true_label == 0] = '#000000' # black
    color_labels.iloc[false_labels == 1] = '#E50000' # red
    color_labels.iloc[false_labels == -1] = '#7E1E9C' # purple

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
        ax[i //3, i %3].scatter(np_compo_features[:,0], np_compo_features[:,1], c = color_labels)
    fig.tight_layout()
    plt.show()

def pairwise_plots_pred_true_thresh(df : pd.DataFrame,
                                    predicted_label : npt.ArrayLike,
                                    true_label : npt.ArrayLike,
                                    axis_thresh : float | npt.ArrayLike):
    """Plots the chosen dataframe after the transformation, pojected onto two dimension

    True positives: green
    False positivs: red
    True negatives: back
    False negatives: purple

    Args:
        df (pd.DataFrame): points to be plotted
        predicted_label (npt.ArrayLike): labels predicted by some algorithm
        true_label (npt.ArrayLike): labels according to "ground truth"
        axis_thresh (float | npt.ArrayLike): theshholds where the distinction was done.
    """

    # convert to numpy
    predicted_label = np.array(predicted_label)
    true_label = np.array(true_label)
    
    if axis_thresh is float:
        axis_thresh = np.ones(df_features.shape[1]) * axis_thresh

    # check least requrements
    assert true_label.shape == predicted_label.shape
    assert true_label.shape[0] == df.shape[0]
    assert axis_thresh.shape[0] == df.shape[1]

    # fist six columns are features
    df_features = df.iloc[:, :6]
    feature_names = df_features.columns
    
    # name thresholds for convenience
    df_axis_thresh = pd.DataFrame(data=axis_thresh.reshape(1,-1), columns=feature_names)
    
    # false positives = 1, false negatives = -1
    false_labels = predicted_label - true_label
    
    # create colors
    color_labels = pd.Series(true_label, copy=True, dtype=str)
    color_labels.iloc[true_label == 1] = '#15B01A' # green
    color_labels.iloc[true_label == 0] = '#000000' # black
    color_labels.iloc[false_labels == 1] = '#E50000' # red
    color_labels.iloc[false_labels == -1] = '#7E1E9C' # purple

    # poltting
    combinations = itertools.combinations(feature_names, 2)
    
    fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    # iterate over combinations for subplots
    for i, combination in enumerate(combinations): 
        df_combo_features = df_features.loc[:, combination]
        np_compo_features = df_combo_features.to_numpy()
        x_feature = combination[0]
        y_feature = combination[1]
        
        ax[i //3, i %3].set_xlabel(x_feature)
        ax[i //3, i %3].set_ylabel(y_feature)
        ax[i //3, i %3].scatter(np_compo_features[:,0], np_compo_features[:,1], c = color_labels)
        ax[i //3, i %3].axhline(y = df_axis_thresh.loc[0,y_feature], c="#000000", linestyle='-')
        ax[i //3, i %3].axvline(x = df_axis_thresh.loc[0,x_feature], c="#000000", linestyle='-')
    fig.tight_layout()
    plt.show()
    
def pairwise_plots_label(df : pd.DataFrame, label : np.array = None):
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