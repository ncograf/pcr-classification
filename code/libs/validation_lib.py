import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, List, Tuple
from itertools import combinations
import transform_lib
from icecream import ic

def validate_labels(df_true_labels : pd.DataFrame,
                    df_predicted_labels : pd.DataFrame,
                    mask : npt.ArrayLike = None,
                    threshold=0.5,
                    verbosity = 1):
    """Print some statistics such as false negatives / positives

    Args:
        true_labels (DataFrame): Ground truth to determine statistics
        predicted_labels (DataFrame): Predictions based on some algorithm can as well be probabilites 
            (in this case we count contributions of variables)
        mask (array_like, optional): Selection of points. Defaults to None.
        threshold (float, optional): Probability threshold in (0,1) above which to consider a sample positively classified. Defautls to 0.5.
        verbosity (int, optional): 0 -> balanced_accuracy, 1 -> stats, 2 -> stats + number of points in classes . Defaults to 0.
    """
    
    if mask is None:
        mask = np.ones(df_predicted_labels.shape[0], dtype=bool)
    
    df_true_labels = df_true_labels.loc[:,df_predicted_labels.columns]
    
    np_true_labels = np.array(df_true_labels.iloc[mask, :])
    np_predicted_labels = np.array(df_predicted_labels.iloc[mask, :])
    
    assert np_true_labels.shape == np_predicted_labels.shape
    assert (df_true_labels.columns == df_predicted_labels.columns).all()
    
    # errors: [1,0] false positive, [-1,0] false negative
    error_matrix = (np_predicted_labels - np_true_labels)
    
    # get outliers
    outlier_mask = (np_predicted_labels < 0)[:,0]
    outlier_rate = np.mean(outlier_mask)
    n_outlier = np.sum(outlier_mask)
    error_matrix = error_matrix[np.logical_not(outlier_mask),:]

    np_true_labels_no_out = np_true_labels[~outlier_mask,:]
    np_pred_labels_no_out = np_predicted_labels[~outlier_mask,:]
    
    # n_points
    n_points = np.sum(np.logical_not(outlier_mask))
    
    # compute total statistics
    abs_error_matrix = np.abs(error_matrix)
    abs_error_rate = np.mean(abs_error_matrix)
    abs_error_rate_class = np.mean(abs_error_matrix, axis=0).reshape(1,-1)
    df_abs_error_rate_class = pd.DataFrame(data=abs_error_rate_class, columns=df_predicted_labels.columns)
    
    # true negatives (we only consider negatives in ground truth)
    true_negatives =  (np_true_labels_no_out == 0) & (np_pred_labels_no_out < threshold)
    assert true_negatives.shape == error_matrix.shape
    score_true_neg = 1 - np_pred_labels_no_out.copy()
    score_true_neg[~true_negatives] = 0
    n_true_negatives = np.sum(score_true_neg)
    n_true_neg_class = np.sum(score_true_neg, axis=0).reshape(1,-1)
    df_ture_neg_class = pd.DataFrame(data=n_true_neg_class, columns=df_predicted_labels.columns)

    # false negatives
    false_negatives = (np_true_labels_no_out == 1) & (np_pred_labels_no_out < threshold) 
    assert false_negatives.shape == error_matrix.shape
    score_false_neg = 1 - np_pred_labels_no_out.copy()
    score_false_neg[~false_negatives] = 0
    n_false_negatives = np.sum(score_false_neg)
    n_false_neg_class = np.sum(score_false_neg, axis=0).reshape(1,-1)
    df_false_neg_class = pd.DataFrame(data=n_false_neg_class, columns=df_predicted_labels.columns)
    
    # true positives
    true_positives = (np_true_labels_no_out == 1) & (np_pred_labels_no_out >= threshold)
    assert true_positives.shape == error_matrix.shape
    score_true_pos = np_pred_labels_no_out.copy()
    score_true_pos[~true_positives] = 0
    n_true_positives = np.sum(score_true_pos)
    n_true_pos_class = np.sum(score_true_pos, axis=0).reshape(1,-1)
    df_true_pos_class = pd.DataFrame(data=n_true_pos_class, columns=df_predicted_labels.columns)
    
    # false positives
    false_positives = (np_true_labels_no_out == 0) & (np_pred_labels_no_out >= threshold)
    assert false_positives.shape == error_matrix.shape
    score_false_pos = np_pred_labels_no_out.copy()
    score_false_pos[~false_positives] = 0
    n_false_positives = np.sum(score_false_pos)
    n_false_pos_class = np.sum(score_false_pos, axis=0).reshape(1,-1)
    df_false_pos_class = pd.DataFrame(data=n_false_pos_class, columns=df_predicted_labels.columns)

    # stats
    precision = (n_true_positives) / (n_true_positives + n_false_positives + 1e-15)
    precision_class = n_true_pos_class / (n_true_pos_class + n_false_pos_class + 1e-15)
    df_precision_class = pd.DataFrame(data=precision_class, columns=df_predicted_labels.columns)

    recall = (n_true_positives) / (n_true_positives + n_false_negatives + 1e-15) # = sensitivey / TPR
    recall_class = n_true_pos_class / (n_true_pos_class + n_false_neg_class + 1e-15)
    df_recall_class = pd.DataFrame(data=recall_class, columns=df_predicted_labels.columns)
    
    specificity = (n_true_negatives) / (n_true_negatives + n_false_positives + 1e-15) # = selectivity / TNR
    specificity_class = n_true_neg_class / (n_true_neg_class + n_false_pos_class + 1e-15)
    df_specificity_class = pd.DataFrame(data=specificity_class, columns=df_predicted_labels.columns)

    accuracy = (n_true_positives + n_true_negatives) / (n_false_positives + n_true_positives + n_false_negatives + n_true_negatives + 1e-15)
    accuracy_class = (n_true_pos_class + n_true_neg_class) / (n_false_pos_class + n_true_pos_class + n_false_neg_class + n_true_neg_class + 1e-15)
    df_accuracy_class = pd.DataFrame(data=accuracy_class, columns=df_predicted_labels.columns)

    balanced_accuracy = (specificity + recall) / 2
    balanced_accuracy_class = (specificity_class + recall_class) / 2
    df_balanced_accuracy_class = pd.DataFrame(data=balanced_accuracy_class, columns=df_predicted_labels.columns)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class + 1e-15)
    df_f1_class = pd.DataFrame(data=f1_class, columns=df_predicted_labels.columns)

    
    print(f'Outlier rate: {outlier_rate}, n_outlier: {n_outlier}\n')
    if(verbosity >= 2):
        print(f'Total Total Points: {n_points}\n')
        print(f'Total TN (True Negatives): {n_true_negatives}\n{df_ture_neg_class.to_string(index=False)}\n')
        print(f'Total TP (True Positives): {n_true_positives}\n{df_true_pos_class.to_string(index=False)}\n')
        print(f'Total FN (False Negatives): {n_false_negatives}\n{df_false_neg_class.to_string(index=False)}\n')
        print(f'Total FP (False Positives): {n_false_positives}\n{df_false_pos_class.to_string(index=False)}\n')
    if(verbosity >= 1):
        print(f'Total error rate: {abs_error_rate}\n{df_abs_error_rate_class.to_string(index=False)}\n')
        print(f'Precision (TP / (TP + FP)): {precision}\n{df_precision_class.to_string(index=False)}\n')
        print(f'Recall / TPR (TP / (TP + FN)): {recall}\n{df_recall_class.to_string(index=False)}\n')
        print(f'Specificity / TNR (TN / (TN + FP)): {specificity}\n{df_specificity_class.to_string(index=False)}\n')
        print(f'Accuracy ((TP + TN) / (P + N)): {accuracy}\n{df_accuracy_class.to_string(index=False)}\n')
        print(f'F1 (2 * (precision * recall) / (precision + recall)): {f1}\n{df_f1_class.to_string(index=False)}\n')
    print(f'Balanced Accuracy (Specificity + Recall) / 2: {balanced_accuracy}\n{df_balanced_accuracy_class.to_string(index=False)}\n')

    return false_negatives, false_positives


def get_false_clusters(clusters : Dict[int, transform_lib.Cluster],
                       df_true_labels : pd.DataFrame,
                       df_predictions : pd.DataFrame,
                       disease : str | List[str],
                       threshold : float = 1,
                       mask : npt.ArrayLike = None,
                       ) -> List[int]:
    """Compute the indices of the falsely classified points

    Args:
        cluster_mask (Dict[npt.ArrayLike]): List of masks for each cluster
        df_true_labels (pd.DataFrame): Ture labels
        df_predictions (pd.DataFrame): False labels
        disease (str | List[str]): Disease indicators
        threshold (float): below what precentage of misspredictions we want the cluster to be displayed
        mask (array_like, optional): Mask for filtering `df_true_labels` and `df_predictions`

    Returns:
        List[int]: Clusters which are falsely classified
    """
    if isinstance(disease, str):
        disease = [disease]

    false_list = []
    false_rate = []

    if mask is None:
        mask = np.ones(df_true_labels.shape[0], dtype=bool)
    
    df_true_labels = df_true_labels.iloc[mask,:]
    df_predictions = df_predictions.iloc[mask,:]
    
    for dim in disease:
        for cluster in clusters.keys():
            true_cluster_labels = df_true_labels.loc[clusters[cluster].mask, dim]
            predicted_label = df_predictions.loc[clusters[cluster].mask, dim]
            n_cluster = true_cluster_labels.shape[0]
            assert true_cluster_labels.shape == predicted_label.shape
            n_true = np.sum(np.array(true_cluster_labels, dtype=bool) == np.array(predicted_label, dtype=bool))
            if n_true / n_cluster < threshold:
                false_rate.append(n_cluster - n_true)
                false_list.append(cluster)
                ic(cluster, dim, n_cluster - n_true)
    
    order = np.argsort(false_rate)
    
    return np.unique(np.array(false_list)[order])

def get_false_cluster_for_plotting(df_data_points : pd.DataFrame,
                                   df_predictions : pd.DataFrame,
                                   df_ground_truth : pd.DataFrame,
                                   clusters : Dict[int, transform_lib.Cluster],
                                   false_clusters : List[int],
                                   mask : npt.ArrayLike = None) -> Tuple[pd.DataFrame]:
    """Extract only the falsely labelled data

    Args:
        df_data_points (pd.DataFrame): All the dataframe points
        df_predictions (pd.DataFrame): All the predictions
        df_ground_truth (pd.DataFrame): All ground truth data
        clusters (Dict[int, Clusters]): Cluster masks for each cluster
        false_clusters (List[int]): List of clusters to exract
        mask (npt.ArrayLike, optional): Selection of points. Defaults to None.

    Returns:
        Tuple[pd.DataFrame]: data to be plottet, prediction labels for data, and ground truth
    """
    if mask is None:
        mask = np.ones(df_data_points.shape[0], dtype=bool)

    np_data = np.concatenate([df_data_points.iloc[np.logical_and(mask, clusters[k].mask),:] for k in false_clusters])
    np_predicions = np.concatenate([df_predictions.iloc[np.logical_and(mask, clusters[k].mask),:] for k in false_clusters])
    np_ground_truth = np.concatenate([df_ground_truth.iloc[np.logical_and(mask, clusters[k].mask),:] for k in false_clusters])
    data = pd.DataFrame(data=np_data, columns=df_data_points.columns)
    predictions = pd.DataFrame(data=np_predicions, columns=df_predictions.columns)
    ground_truth = pd.DataFrame(data=np_ground_truth, columns=df_ground_truth.columns)

    return data, predictions, ground_truth
    
def validate_combinations(df_true_labels : pd.DataFrame, df_predicted_labels : pd.DataFrame, mask : npt.ArrayLike = None,
                          threshold : float = 0.5, verbosity = 0):
    """Print some statistics for all combinations of diseases

    Args:
        true_labels (DataFrame): Ground truth to determine statistics
        predicted_labels (DataFrame): Predictions based on some algorithm
        mask (array_like, optional): Selection of points. Defaults to None.
        verbosity (int, optional): 0 -> balanced_accuracy, 1 -> stats, 2 -> stats + number of points in classes . Defaults to 0.
    """
    
    if mask is None:
        mask = np.ones(df_predicted_labels.shape[0], dtype=bool)
    
    df_true_labels = df_true_labels.loc[:,df_predicted_labels.columns]
    
    np_true_labels = np.array(df_true_labels.iloc[mask, :])
    np_predicted_labels = np.array(df_predicted_labels.iloc[mask, :])
    
    assert np_true_labels.shape == np_predicted_labels.shape
    assert (df_true_labels.columns == df_predicted_labels.columns).all()
    
    n_labels = df_predicted_labels.shape[1]
    label_combinations = []
    for sub in range(1,n_labels+1):
        label_combinations.extend(combinations(df_predicted_labels.columns, sub))
    
    # compute outlier
    outlier_mask = (np_predicted_labels < 0)[:,0]
    n_outlier = np.sum(outlier_mask)

    print(f'Total number of outlayer: {n_outlier}\n')

    for labels in label_combinations:
        
        not_labels = df_predicted_labels.columns.difference(labels)
        
        true_sel = np.array(df_true_labels.loc[:, labels],dtype=bool)
        true_not_sel = np.array(df_true_labels.loc[:, not_labels], dtype=bool)
        pred_sel = np.array(df_predicted_labels.loc[:, labels],dtype=bool)
        pred_not_sel = np.array(df_predicted_labels.loc[:, not_labels], dtype=bool)
        
        true_mask = np.logical_and(np.all(true_sel, axis=1),np.all(np.logical_not(true_not_sel), axis=1))
        pred_mask = np.logical_and(np.all(pred_sel, axis=1),np.all(np.logical_not(pred_not_sel), axis=1))

        # get outliers
        outlier_mask_local = (np_predicted_labels < 0)[true_mask,0]
        
        pred_mask = pred_mask[np.logical_not(outlier_mask)]
        true_mask = true_mask[np.logical_not(outlier_mask)]

        n_outlier_local = np.sum(outlier_mask_local)
        
        # ture positives
        n_true_pos = np.sum(np.logical_and(pred_mask, true_mask))
        n_true_neg = np.sum(np.logical_and(np.logical_not(pred_mask), np.logical_not(true_mask)))
        n_false_pos = np.sum(np.logical_and(pred_mask, np.logical_not(true_mask)))
        n_false_neg = np.sum(np.logical_and(np.logical_not(pred_mask), true_mask))
        
        # stats
        if n_true_pos + n_false_pos != 0:
            recall = (n_true_pos) / (n_true_pos + n_false_neg + 1e-15) # = sensitivey / TPR
        else:
            recall = "No positives"

        if n_true_pos + n_false_pos != 0:
            precision = (n_true_pos) / (n_true_pos + n_false_pos + 1e-15)
        else:
            precision = "No pos predictions"
        
        if n_true_neg + n_false_pos != 0:
            specificity = (n_true_neg) / (n_true_neg + n_false_pos + 1e-15) # = selectivity / TNR
        else:
            specificity = "No negatives"

        accuracy = (n_true_pos + n_true_neg) / (n_false_pos + n_true_pos + n_false_neg + n_true_neg + 1e-15)

        if isinstance(specificity, str) or isinstance(recall, str):
            balanced_accuracy = "No evaluation possible"
        else:
            balanced_accuracy = (specificity + recall) / 2

        if isinstance(precision,str) or isinstance(recall,str):
            f1 = "No evaluation possible"
        else:
            f1 = 2 * (precision * recall) / (precision + recall)


    
        print(f'Label combination {labels}:')
        print(f'n outliers: {n_outlier_local}, n_true_pos: {n_true_pos}')
        if(verbosity >= 2):
            print(f'Total TN (True Negatives): {n_true_neg}')
            print(f'Total TP (True Positives): {n_true_pos}')
            print(f'Total FN (False Negatives): {n_false_neg}')
            print(f'Total FP (False Positives): {n_false_pos}')
        if(verbosity >= 1):
            print(f'Precision (TP / (TP + FP)): {precision}')
            print(f'Recall / TPR (TP / (TP + FN)): {recall}')
            print(f'Specificity / TNR (TN / (TN + FP)): {specificity}')
            print(f'Accuracy ((TP + TN) / (P + N)): {accuracy}')
            print(f'F1 (2 * (precision * recall) / (precision + recall)): {f1}')
        print(f'Balanced Accuracy: {balanced_accuracy}\n')

    return

def get_negative_percent(gt : pd.DataFrame, axis : List[str]):
    
    np_points = np.array(gt.loc[:,axis],dtype=bool)
    negative_count = np.sum(np_points, axis=0)
    negative_percent = negative_count / np_points.shape[0]
    ic(negative_count.shape)
    ic(negative_percent.shape)
    ic(axis)
    df_negative_count = pd.DataFrame(data=np.reshape(negative_count, (1,-1)), columns=axis)
    df_negative_percent = pd.DataFrame(data=np.reshape(negative_percent, (1,-1)), columns=axis)
    print(f'Positive count: \n{df_negative_count}')
    print(f'Positive percent: \n{df_negative_percent}')
    print(f'Negative percent: \n{1 - df_negative_percent}')
    
    all_neg = np.sum(np.all(~(np_points.astype(bool)),axis=1))
    all_neg_per = all_neg / np_points.shape[0]
    ic(all_neg.shape)
    ic(all_neg_per.shape)
    ic(np.var(np_points, axis=0))
    print(f'All neg Count count: \n{all_neg}')
    print(f'All neg percent: \n{all_neg_per}')
    print(f'All pos percent: \n{1 - all_neg_per}')
    