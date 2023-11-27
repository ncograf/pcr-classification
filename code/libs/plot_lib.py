import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from icecream import ic
from typing import Tuple, List

def pairwise_plots_pred_true(df : pd.DataFrame,
                                predicted_label : npt.ArrayLike,
                                true_label : npt.ArrayLike,
                                axis_thresh : float | npt.ArrayLike = None):
    """Plots the chosen dataframe, pojected onto two dimensions for all possible combinations
    of tow dimensions. Optionally threshholds are also drawn.
    
    Note that in case threshholds are used, it is important that the scale of df matches the
    scale in which the threshholds are applied.

    True positives: green
    False positivs: red
    True negatives: back
    False negatives: purple

    Args:
        df (pd.DataFrame): points to be plotted
        predicted_label (npt.ArrayLike): labels predicted by some algorithm
        true_label (npt.ArrayLike): labels according to "ground truth"
        axis_thresh (float | npt.ArrayLike, optional): theshholds where the distinction was done. Defaults to None.
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
    if not axis_thresh is None:
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
        if not axis_thresh is None:
            ax[i //3, i %3].axhline(y = df_axis_thresh.loc[0,y_feature], c="#000000", linestyle='-')
            ax[i //3, i %3].axvline(x = df_axis_thresh.loc[0,x_feature], c="#000000", linestyle='-')
    fig.tight_layout()
    plt.show()
    
def pairwise_plots_label(df : pd.DataFrame, label : np.array = None):

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
    

def plot_pairwise_selection(
                            data_points : pd.DataFrame,
                            predictions : pd.DataFrame,
                            ground_trouth : pd.DataFrame,
                            selected_pairs : List[Tuple[str, str]],
                            axis_thresh : pd.DataFrame = None,
                            n_cols : int = 2,
                            mask : npt.ArrayLike = None,
                            ):
    
    # check least requrements
    for (color_one, color_two) in selected_pairs:

        assert color_two in data_points.columns
        assert color_one in data_points.columns

        assert color_two in predictions.columns
        assert color_one in predictions.columns

        assert color_two in ground_trouth.columns
        assert color_one in ground_trouth.columns

        if not axis_thresh  is None:
            assert color_two in axis_thresh.columns
            assert color_one in axis_thresh.columns


    assert ground_trouth.shape[0] == predictions.shape[0]
    assert ground_trouth.shape[0] == data_points.shape[0]

    if mask is None:
        mask = np.ones(data_points.shape[0], dtype=bool)
    # --------
    # plotting
    # --------
    
    # check the number of plots to be created
    n = len(selected_pairs)
    n_rows = n // n_cols + 0 if n % n_cols == 0 else 1
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=False, sharey=False)
    fig.set_figheight(5 * n_rows)
    fig.set_figwidth(7 * n_cols)
    
    # iterate over combinations for subplots
    for i, (col_one, col_two) in enumerate(selected_pairs): 

        # false positives = 1, false negatives = -1
        true_labels = ground_trouth.loc[mask,col_one].to_numpy()
        pred_labels = predictions.loc[mask,col_one].to_numpy()
        false_labels = pred_labels - true_labels
        
        # create colors
        color_labels = pd.Series(true_labels, copy=True, dtype=str)
        color_labels.iloc[true_labels == 1] = '#15B01A' # green
        color_labels.iloc[true_labels == 0] = '#000000' # black
        color_labels.iloc[false_labels == 1] = '#E50000' # red
        color_labels.iloc[false_labels == -1] = '#7E1E9C' # purple
        
        # get outliers
        color_labels.iloc[pred_labels < 0] = '#FFFF33'  # ugly yellow
        
        
        x_features = data_points.loc[mask, col_one].to_numpy()
        y_features = data_points.loc[mask, col_two].to_numpy()

        ax[i // n_cols, i % n_cols].set_xlabel(col_one)
        ax[i // n_cols, i % n_cols].set_ylabel(col_two)
        ax[i // n_cols, i % n_cols].scatter(x_features, y_features, c = color_labels)
        if not axis_thresh is None:
            ax[i // n_cols, i % n_cols].axvline(x = axis_thresh.loc[0,col_one], c="#000000", linestyle='-')
    fig.tight_layout()
    plt.show()
    
def plot_pairwise_selection_bayesian(
                            data_points : pd.DataFrame,
                            predictions : pd.DataFrame,
                            ground_trouth : pd.DataFrame,
                            selected_pairs : List[Tuple[str, str]],
                            axis_thresh : pd.DataFrame = None,
                            n_cols : int = 2,
                            mask : npt.ArrayLike = None,
                            title : str = None,
                            save_path : str = None
                            ):
    
    # check least requrements
    for (color_one, color_two) in selected_pairs:

        assert color_two in data_points.columns
        assert color_one in data_points.columns

        assert color_two in predictions.columns
        assert color_one in predictions.columns

        assert color_two in ground_trouth.columns
        assert color_one in ground_trouth.columns

        if not axis_thresh  is None:
            assert color_two in axis_thresh.columns
            assert color_one in axis_thresh.columns


    assert ground_trouth.shape[0] == predictions.shape[0]
    assert ground_trouth.shape[0] == data_points.shape[0]

    if mask is None:
        mask = np.ones(data_points.shape[0], dtype=bool)

    # --------
    # plotting
    # --------
    
    # check the number of plots to be created
    n = len(selected_pairs)
    n_rows = n // n_cols + 0 if n % n_cols == 0 else 1
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=False, sharey=False)
    fig.set_figheight(5 * n_rows)
    fig.set_figwidth(7 * n_cols)
    
    # iterate over combinations for subplots
    for i, (col_one, col_two) in enumerate(selected_pairs): 

        # false positives = 1, false negatives = -1
        true_labels = ground_trouth.loc[mask,col_one].to_numpy()
        pred_labels = predictions.loc[mask,col_one].to_numpy()
        
        # create colors
        green = np.array((21, 176, 26), dtype=np.float32) / 255  # green #15B01A
        green_value = np.einsum("i,j->ji", green, pred_labels)
        black = np.array((0,0,0),dtype=np.float32) / 255 # black #000000
        black_value = np.einsum("i,j->ji",black,  (1 - pred_labels))
        red = np.array((229,0,0)) / 255 # red #E50000
        red_value = np.einsum("i,j->ji",red, pred_labels) 
        purple = np.array((126, 30 ,156), np.float32) / 255 # purple #7E1E9C
        purple_value = np.einsum("i,j->ji",purple,  (1 - pred_labels))
        positive_contributions = np.einsum("jc,j->jc", green_value + purple_value, (true_labels == 1))
        negative_contributions = np.einsum("jc,j->jc", red_value + black_value, (true_labels == 0))
        color_labels = negative_contributions + positive_contributions

        # get outliers
        color_labels[pred_labels < 0] = np.array((1,1,0)) #'FFFF33'  ugly yellow
        
        x_features = data_points.loc[mask, col_one].to_numpy()
        y_features = data_points.loc[mask, col_two].to_numpy()

        ax[i // n_cols, i % n_cols].set_xlabel(col_one)
        ax[i // n_cols, i % n_cols].set_ylabel(col_two)
        ax[i // n_cols, i % n_cols].scatter(x_features, y_features, c = color_labels)
        if not axis_thresh is None:
            ax[i // n_cols, i % n_cols].axvline(x = axis_thresh.loc[0,col_one], c="#000000", linestyle='-')
    fig.tight_layout()
    if not title is None:
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title, fontsize=24)
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_pairwise_selection_label(data_points : pd.DataFrame, 
                                  selected_pairs : List[Tuple[str, str]],
                                  label : np.array = None,
                                  n_cols : int = 2,
                                  ):

    # check the number of plots to be created
    n = len(selected_pairs)
    n_rows = n // n_cols + 0 if n % n_cols == 0 else 1
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=False, sharey=False)
    fig.set_figheight(5 * n_rows)
    fig.set_figwidth(7 * n_cols)
    
    # iterate over combinations for subplots
    for i, (col_one, col_two) in enumerate(selected_pairs): 

        # false positives = 1, false negatives = -1
        x_features = data_points.loc[:, col_one].to_numpy()
        y_features = data_points.loc[:, col_two].to_numpy()

        ax[i // n_cols, i % n_cols].set_xlabel(col_one)
        ax[i // n_cols, i % n_cols].set_ylabel(col_two)
        ax[i // n_cols, i % n_cols].scatter(x_features, y_features, c = label)

    fig.tight_layout()
    plt.show()