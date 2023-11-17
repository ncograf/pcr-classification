import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, List, Tuple
from icecream import ic

def validate_labels(df_true_labels : pd.DataFrame, df_predicted_labels : pd.DataFrame, mask : npt.ArrayLike = None):
    """Print some statistics such as false negatives / positives

    Args:
        true_labels (DataFrame): Ground truth to determine statistics
        predicted_labels (DataFrame): Predictions based on some algorithm
        mask (array_like, optional): Mask to filter e.g. outlier. Defaults to None.
    """
    
    if mask is None:
        mask = np.ones(df_true_labels.shape[0], dtype=bool)
    
    df_true_labels = df_true_labels.loc[:,df_predicted_labels.columns]
    
    np_true_labels = np.array(df_true_labels.iloc[mask,:])
    np_predicted_labels = np.array(df_predicted_labels.iloc[mask,:])
    
    assert np_true_labels.shape == np_predicted_labels.shape
    assert (df_true_labels.columns == df_predicted_labels.columns).all()
    
    # errors: 1 false positive, -1 false negative
    error_matrix = np_predicted_labels - np_true_labels
    
    # get outliers
    outlier_mask = (np_predicted_labels < 0)[:,0]
    outlier_rate = np.mean(outlier_mask)
    n_outlier = np.sum(outlier_mask)
    error_matrix = error_matrix[np.logical_not(outlier_mask),:]
    np_true_labels_no_outliers = np_true_labels[np.logical_not(outlier_mask),:]
    
    # compute total statistics
    abs_error_matrix = np.abs(error_matrix)
    abs_error_rate = np.mean(abs_error_matrix)
    abs_error_rate_class = np.mean(abs_error_matrix, axis=0).reshape(1,-1)
    df_abs_error_rate_class = pd.DataFrame(data=abs_error_rate_class, columns=df_predicted_labels.columns)
    
    # true negatives
    true_negatives = np.logical_and(error_matrix == 0, np_true_labels_no_outliers == 0)
    n_true_negatives = np.sum(true_negatives)
    n_true_neg_class = np.sum(true_negatives, axis=0).reshape(1,-1)

    # false negatives
    false_negatives = error_matrix == -1
    n_false_negatives = np.sum(false_negatives)
    n_false_neg_class = np.sum(false_negatives, axis=0).reshape(1,-1)
    
    # true positives
    true_positives = np.logical_and(error_matrix == 0, np_true_labels_no_outliers == 1)
    n_true_positives = np.sum(true_positives)
    n_true_pos_class = np.sum(true_positives, axis=0).reshape(1,-1)
    
    # false positives
    false_positives = error_matrix == 1
    n_false_positives = np.sum(false_positives)
    n_false_pos_class = np.sum(false_positives, axis=0).reshape(1,-1)

    # stats
    precision = (n_true_positives) / (n_true_positives + n_false_positives)
    precision_class = n_true_pos_class / (n_true_pos_class + n_false_pos_class)
    df_precision_class = pd.DataFrame(data=precision_class, columns=df_predicted_labels.columns)

    recall = (n_true_positives) / (n_true_positives + n_false_negatives) # = sensitivey / TPR
    recall_class = n_true_pos_class / (n_true_pos_class + n_false_neg_class)
    df_recall_class = pd.DataFrame(data=recall_class, columns=df_predicted_labels.columns)
    
    specificity = (n_true_negatives) / (n_true_negatives + n_false_positives) # = selectivity / TNR
    specificity_class = n_true_neg_class / (n_true_neg_class + n_false_pos_class)
    df_specificity_class = pd.DataFrame(data=specificity_class, columns=df_predicted_labels.columns)

    accuracy = (n_true_positives + n_true_negatives) / (n_false_positives + n_true_positives + n_false_negatives + n_true_negatives)
    accuracy_class = (n_true_pos_class + n_true_neg_class) / (n_false_pos_class + n_true_pos_class + n_false_neg_class + n_true_neg_class)
    df_accuracy_class = pd.DataFrame(data=accuracy_class, columns=df_predicted_labels.columns)

    balanced_accuracy = (specificity + recall) / 2
    balanced_accuracy_class = (specificity_class + recall_class) / 2
    df_balanced_accuracy_class = pd.DataFrame(data=balanced_accuracy_class, columns=df_predicted_labels.columns)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)
    df_f1_class = pd.DataFrame(data=f1_class, columns=df_predicted_labels.columns)

    
    print(f'Outlier rate: {outlier_rate}, n_outlier: {n_outlier}\n')
    print(f'Total error rate: {abs_error_rate}\n{df_abs_error_rate_class.to_string(index=False)}\n')
    print(f'Precision (TP / (TP + FP)): {precision}\n{df_precision_class.to_string(index=False)}\n')
    print(f'Recall / TPR (TP / (TP + FN)): {recall}\n{df_recall_class.to_string(index=False)}\n')
    print(f'Specificity / TNR (TN / (TN + FP)): {specificity}\n{df_specificity_class.to_string(index=False)}\n')
    print(f'Accuracy ((TP + TN) / (P + N)): {accuracy}\n{df_accuracy_class.to_string(index=False)}\n')
    print(f'Balanced Accuracy (Specificity + Recall) / 2: {balanced_accuracy}\n{df_balanced_accuracy_class.to_string(index=False)}\n')
    print(f'F1 (2 * (precision * recall) / (precision + recall)): {f1}\n{df_f1_class.to_string(index=False)}\n')



def get_false_clusters(cluster_mask : Dict[int, npt.ArrayLike],
                       df_true_labels : pd.DataFrame,
                       df_predictions : pd.DataFrame,
                       disease : str | List[str],
                       mask : npt.ArrayLike = None) -> List[int]:
    """Compute the indices of the falsely classified points

    Args:
        cluster_mask (Dict[npt.ArrayLike]): List of masks for each cluster
        df_true_labels (pd.DataFrame): Ture labels
        df_predictions (pd.DataFrame): False labels
        disease (str | List[str]): Disease indicators
        mask (array_like, optional): Mask for filtering `df_true_labels` and `df_predictions`

    Returns:
        List[int]: Clusters which are falsely classified
    """
    if isinstance(disease, str):
        disease = [disease]

    false_list = []

    if mask is None:
        mask = np.ones(df_true_labels.shape[0], dtype=bool)
    
    df_true_labels = df_true_labels.iloc[mask,:]
    df_predictions = df_predictions.iloc[mask,:]
    
    for dim in disease:
        for cluster in cluster_mask.keys():
            true_cluster_labels = df_true_labels.loc[cluster_mask[cluster], dim]
            predicted_label = df_predictions.loc[cluster_mask[cluster], dim]
            n_cluster = true_cluster_labels.shape[0]
            assert true_cluster_labels.shape == predicted_label.shape
            n_true = np.sum(np.array(true_cluster_labels) == np.array(predicted_label))
            if n_true / n_cluster < 1:
                false_list.append(cluster)
    
    return false_list

def get_false_cluster_for_plotting(df_data_points : pd.DataFrame,
                                   df_predictions : pd.DataFrame,
                                   df_ground_truth : pd.DataFrame,
                                   cluster_masks : Dict[int, npt.ArrayLike],
                                   false_clusters : List[int],
                                   all_mask : npt.ArrayLike = None) -> Tuple[pd.DataFrame]:
    """Extract only the falsely labelled data

    Args:
        df_data_points (pd.DataFrame): All the dataframe points
        df_predictions (pd.DataFrame): All the predictions
        df_ground_truth (pd.DataFrame): All ground truth data
        cluster_masks (Dict[npt.ArrayLike]): Cluster masks for each cluster
        false_clusters (List[int]): List of clusters to exract
        all_mask (npt.ArrayLike, optional): Selection of points, usually to fit the cluster masks. Defaults to None.

    Returns:
        Tuple[pd.DataFrame]: _description_
    """
    if all_mask is None:
        all_mask = np.ones(df_data_points.shape[0], dtype=bool)

    np_data = np.concatenate([df_data_points.iloc[all_mask,:].iloc[cluster_masks[k]] for k in false_clusters])
    np_predicions = np.concatenate([df_predictions.iloc[all_mask,:].iloc[cluster_masks[k]] for k in false_clusters])
    np_ground_truth = np.concatenate([df_ground_truth.iloc[all_mask,:].iloc[cluster_masks[k]] for k in false_clusters])
    data = pd.DataFrame(data=np_data, columns=df_data_points.columns)
    predictions = pd.DataFrame(data=np_predicions, columns=df_predictions.columns)
    ground_truth = pd.DataFrame(data=np_ground_truth, columns=df_ground_truth.columns)

    return data, predictions, ground_truth
    