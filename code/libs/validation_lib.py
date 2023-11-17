import numpy as np
import pandas as pd
from icecream import ic

def validate_labels(df_true_labels : pd.DataFrame, df_predicted_labels : pd.DataFrame):
    """Print some statistics such as false negatives / positives

    Args:
        true_labels (DataFrame): Ground truth to determine statistics
        predicted_labels (DataFrame): Predictions based on some algorithm
    """
    
    df_true_labels = df_true_labels.loc[:,df_predicted_labels.columns]
    
    np_true_labels = np.array(df_true_labels)
    np_predicted_labels = np.array(df_predicted_labels)
    
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
    true_neg_rate = np.mean(true_negatives)
    n_true_negatives = np.sum(true_negatives)
    true_neg_rate_class = np.mean(true_negatives, axis=0).reshape(1,-1)
    n_true_neg_class = np.sum(true_negatives, axis=0).reshape(1,-1)
    df_true_neg_rate_class = pd.DataFrame(data=true_neg_rate_class, columns=df_predicted_labels.columns)

    # false negatives
    false_negatives = error_matrix == -1
    false_neg_rate = np.mean(false_negatives)
    n_false_negatives = np.sum(false_negatives)
    false_neg_rate_class = np.mean(false_negatives, axis=0).reshape(1,-1)
    n_false_neg_class = np.sum(false_negatives, axis=0).reshape(1,-1)
    df_false_neg_rate_class = pd.DataFrame(data=false_neg_rate_class, columns=df_predicted_labels.columns)
    
    # true positives
    true_positives = np.logical_and(error_matrix == 0, np_true_labels_no_outliers == 1)
    true_pos_rate = np.mean(true_positives)
    n_true_positives = np.sum(true_positives)
    true_pos_rate_class = np.mean(true_positives, axis=0).reshape(1,-1)
    n_true_pos_class = np.sum(true_positives, axis=0).reshape(1,-1)
    df_true_pos_rate_class = pd.DataFrame(data=true_pos_rate_class, columns=df_predicted_labels.columns)
    
    # false positives
    false_positives = error_matrix == 1
    false_pos_rate = np.mean(false_positives)
    n_false_positives = np.sum(false_positives)
    false_pos_rate_class = np.mean(false_positives, axis=0).reshape(1,-1)
    n_false_pos_class = np.sum(false_positives, axis=0).reshape(1,-1)
    df_false_pos_rate_class = pd.DataFrame(data=false_pos_rate_class, columns=df_predicted_labels.columns)
    
    # stats
    precision = (n_true_positives) / (n_true_positives + n_false_positives)
    precision_class = n_true_pos_class / (n_true_pos_class + n_false_pos_class)
    df_precision_class = pd.DataFrame(data=precision_class, columns=df_predicted_labels.columns)

    recall = (n_true_positives) / (n_true_positives + n_false_negatives)
    recall_class = n_true_pos_class / (n_true_pos_class + n_false_neg_class)
    df_recall_class = pd.DataFrame(data=recall_class, columns=df_predicted_labels.columns)

    accuracy = (n_true_positives + n_true_negatives) / (n_true_positives + n_true_positives + n_false_negatives + n_true_negatives)
    accuracy_class = (n_true_pos_class + n_true_neg_class) / (n_true_pos_class + n_true_pos_class + n_false_neg_class + n_true_neg_class)
    df_accuracy_class = pd.DataFrame(data=accuracy_class, columns=df_predicted_labels.columns)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)
    df_f1_class = pd.DataFrame(data=f1_class, columns=df_predicted_labels.columns)

    
    print(f'Outlier rate: {outlier_rate}, n_outlier: {n_outlier}\n')
    print(f'Total error rate: {abs_error_rate}\n{df_abs_error_rate_class.to_string(index=False)}\n')
    print(f'False negative rate: {false_neg_rate}\n{df_false_neg_rate_class.to_string(index=False)}\n')
    #print(f'False positive rate: {false_pos_rate}\n{df_false_pos_rate_class.to_string(index=False)}\n')
    print(f'True negative rate: {true_neg_rate}\n{df_true_neg_rate_class.to_string(index=False)}\n')
    #print(f'True positive rate: {true_pos_rate}\n{df_true_pos_rate_class.to_string(index=False)}\n')
    print(f'Precision (TP / (TP + FP)): {precision}\n{df_precision_class.to_string(index=False)}\n')
    print(f'Recall (TP / (TP + FN)): {recall}\n{df_recall_class.to_string(index=False)}\n')
    print(f'Accuracy ((TP + TN) / (P + N)): {accuracy}\n{df_accuracy_class.to_string(index=False)}\n')
    print(f'F1 (2 * (precision * recall) / (precision + recall)): {f1}\n{df_f1_class.to_string(index=False)}\n')
