import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from typing import List
import transform_lib
from icecream import ic

class ThresholdClassifier(BaseEstimator):
    """Classifier, for the list of classes based on
    thresholds in transformed axis and clusterwise classifications
    
    Frist the classifier relies a negative control sample and a positive
    control sample. The negative sample it to determine the region of the 
    negative cluster, in order to adapt the static theshold somewhat dynamically
    on the data, accounting for small changes in the experiment setting.
    
    Then the positive control is needed to determin the transformation
    matirx on the data in order to scale a reasonable amount in
    all dimensions.

    In the  

    Args:
        BaseEstimator : sklearn compatibility
    """
    
    def __init__(self, 
                 negative_control : npt.ArrayLike,
                 positive_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 criterion : str,
                 transform_base : str = "neg",
                 aggressiveness : npt.ArrayLike | float = 10,
                 whitening_transformer : TransformerMixin = transform_lib.WhitenTransformer(transform_lib.Whitenings.ZCA_COR),
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS']):
        """Initialize classifier with important parameters
        
        The negative control is only use for determeining the thresholds of the labels / axes.
        These thesholds are also influenced by the whitening transformer and the aggressiveness.
        
        The cluster algorithm is used to determine clusters. All the points in one cluster will be labelled positive or
        negative depending on it's center. 

        The prediction axis variable makes sure the right desieses get determined by the right axes.
        
        Args:
            negative_control (npt.ArrayLike): Sample points used as negative control
            positive_control (npt.ArrayLike): Sample points used as positive control
            cluster_algorithm (ClusterMixin): Clustering algorithm to be used for builing the clusters
            criterion (str): threshhold criterion e.g. "std_neg", "std_pos", "min_max_all"
            transform_base (str): cluster on which to compute the transform e.g. "neg", "pos", "dynamic"
            aggressiveness (npt.ArrayLike | float, optional): Threshhold points per axis/label based in units of
                standard deviations of the zero cluster in this dimension. Defaults to 10.
            whitening_transformer (TransformerMixin, optional): Whiteining transformer to be used
                in order to eliminate correlation. Defaults to transform_lib.WhitenTransformer(transform_lib.Whitenings.ZCA_COR).
            prediction_axis (List[str], optional): Assigns labels to the prediction axis. 
                Defaults to ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'].
        """
        
        # store local variables
        self.aggressivness = aggressiveness
        self.whitening_transformer = whitening_transformer
        self.prediction_axis = prediction_axis
        self.cluster_algorithm = cluster_algorithm
        self.negative_control = negative_control
        self.positive_control = positive_control
        self.criterion = criterion
        self.transform_base = transform_base
        
        self.axis_threshholds = None
        self.X_fit = None
        self.X_fit_transformed = None
        self.X_transformed = None
        self.cluster_labels = None
        
    def fit(self, X : npt.ArrayLike, y : npt.ArrayLike = None):
        """Determine thresholds based on the input X
        
        The covariance for the Covariance elimination is taken from the sample
        X. This will usually be the same X as used for the prediction, but it
        may also differ.
        
        For example this X can contain all the samples, while
        the input to the predict function is limited in the size of X due to the
        clustering algorithm as bottlneck.

        Args:
            X (npt.ArrayLike): Sample used to compute covariance for whitening.
            y (npt.ArrayLike, optional): Ignored. Defaults to None.

        Returns:
            self
        """
        
        # transform base map
        transform_base_map = {
            "neg" : self.negative_control,
            "pos" : self.positive_control,
            "dynamic" : X
        }
        
        # determine fitting X
        self.X_fit = transform_base_map[self.transform_base]
        
        # create transformed inputs
        self.X_fit_whitened = self.whitening_transformer.fit_transform(self.X_fit)
        self.X_fit_transformed = self.X_fit_whitened
        self.negative_control_transformed = self.whitening_transformer.transform(self.negative_control)
        self.positive_control_transformed = self.whitening_transformer.transform(self.positive_control)
        
        # get threshholds
        self.axis_threshholds = get_per_axis_criterion(self.negative_control_transformed,
                                                       self.positive_control_transformed,
                                                       self.X_fit_transformed,
                                                       self.criterion,
                                                       self.aggressivness)
        
        return self
        
    
    def predict(self, X : npt.ArrayLike,
                y : npt.ArrayLike = None,
                ) -> pd.DataFrame:
        """Assigns a list of true / false indicators for each input in X
        
        This method is limited in the sense that the clustering algorithm might collaps 
        for too large inputs in X.
        
        The labels are determied based on the thresholds computed in fit (which has to be called
        before calling this). Moreover all points in the same cluster get the same labels, 
        according to the clusters mean.

        Args:
            X (npt.ArrayLike): Samples to be classified
            y (npt.ArrayLike, optional): Ignored. Defaults to None.

        Returns:
            pd.DataFrame: Indicators for each label
        """
        
        # transform input
        self.X_whitened = self.whitening_transformer.transform(X)
        self.X_transformed = self.X_whitened
        
        ### get clusters ###
        X_to_cluster = self.X_transformed
        
        # clustering does not scale well TODO
        self.cluster_labels = self.cluster_algorithm.fit_predict(X_to_cluster)

        # make predictions
        predictions = get_cluster_based_classification(self.X_transformed, self.cluster_labels, self.axis_threshholds)
        
        # add labels to predictions
        self.predictions_df = pd.DataFrame(data = predictions, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def validate_labels(self, true_labels : npt.ArrayLike):
        """Print some statistics such as false negatives / positives

        Args:
            true_labels (npt.ArrayLike): Ground truth to determine statistics
        """
        
        np_true_labels = np.array(true_labels)
        np_predicted_labels = np.array(self.predictions_df)
        
        assert np_true_labels.shape == np_predicted_labels.shape
        
        # errors: 1 false positive, -1 false negative
        error_matrix = np_predicted_labels - np_true_labels
        
        # compute total statistics
        abs_error_matrix = np.abs(error_matrix)
        abs_error_rate = np.mean(abs_error_matrix)
        abs_error_rate_class = np.mean(abs_error_matrix, axis=0).reshape(1,-1)
        df_abs_error_rate_class = pd.DataFrame(data=abs_error_rate_class, columns=self.predictions_df.columns)

        # false negatives
        false_negatives = error_matrix == -1
        false_neg_rate = np.mean(false_negatives)
        false_neg_rate_class = np.mean(false_negatives, axis=0).reshape(1,-1)
        df_false_neg_rate_class = pd.DataFrame(data=false_neg_rate_class, columns=self.predictions_df.columns)
        
        # false positives
        false_positives = error_matrix == 1
        false_pos_rate = np.mean(false_positives)
        false_pos_rate_class = np.mean(false_positives, axis=0).reshape(1,-1)
        df_false_pos_rate_class = pd.DataFrame(data=false_pos_rate_class, columns=self.predictions_df.columns)

        
        print(f'Total error rate: {abs_error_rate}\nTotal error per class:\n {df_abs_error_rate_class}\n\n')
        print(f'False negative rate: {false_neg_rate}\nFalse negative rate per class:\n {df_false_neg_rate_class}\n\n')
        print(f'False positive rate: {false_pos_rate}\nFalse negative rate per class:\n {df_false_pos_rate_class}\n\n')
        

def get_per_axis_criterion(neg_control_transformed : npt.ArrayLike,
                           pos_control_transformed : npt.ArrayLike,
                           all_samples_transformed : npt.ArrayLike,
                           criterion : str,
                           aggressiveness : float | npt.ArrayLike = 0.7) -> npt.NDArray:
    """Cluster based criterium for classification. The criterium
    will one threshold per axis in transformed coordinates.
    
    The idea is that in the transformed coordinates, there is no correlation between the axis
    and hence the individual classes / deseases should be separable along exactly one
    dimension via a threshold.

    Threshholds are computed as the mean of the control cluster plus
    aggressiveness times the standard deviation along the corresponding axis.
    
    This is an auxiliary function

    Args:
        neg_control_transformed (array_like): negative control sample in transformed coordinates
        pos_control_transformed (array_like): positive control sample in transformed coordinates
        all_samples_transformed (array_like): all samples in transformed coordinates
        criterion (str) : criterion to base the thresholds on.
        aggressiveness (float): Number of standard deviation to be added to the mean 
            for thresholds (only used for certain criteria given in previous argument). Defaults to 0.7.

    Returns:
        numpy ndarray : One threshold per input column as described above.
    """
    
    # convert to numpy and transform to get meaningful representation
    
    # compute standard deviation along the transformed axis
    std_neg = np.std(neg_control_transformed, axis=0)
    std_pos = np.std(pos_control_transformed, axis=0)
    std_all = np.std(all_samples_transformed, axis=0)
    
    # get mean in transformed coordinates
    mean_neg = np.mean(neg_control_transformed, axis=0)
    mean_pos = np.mean(pos_control_transformed, axis=0)
    mean_all = np.mean(all_samples_transformed, axis=0)
    
    # std criterion
    std_neg_criterion = mean_neg + std_neg * aggressiveness
    std_pos_criterion = mean_pos + std_pos * aggressiveness
    std_all_criterion = mean_all + std_all * aggressiveness

    # compute min and max
    min_neg = np.min(neg_control_transformed, axis=0)
    max_neg = np.max(neg_control_transformed, axis=0)
    min_pos = np.min(pos_control_transformed, axis=0)
    max_pos = np.max(pos_control_transformed, axis=0)
    min_all = np.min(all_samples_transformed, axis=0)
    max_all = np.max(all_samples_transformed, axis=0)
    
    # min-max criterion
    min_max_neg_criterion = (max_neg + min_neg) / 2
    min_max_pos_criterion = (max_pos + min_pos) / 2
    min_max_all_criterion = (max_all + min_all) / 2
    
    # mixed criterion
    # pos minmax to negative std inverse ratio
    ratio = std_neg_criterion / min_max_pos_criterion
    mixed_criterion = ratio * std_neg_criterion + (1-ratio) * min_max_pos_criterion
    
    # generate map
    criterion_map = {
        "std_neg" : std_neg_criterion,
        "std_pos" : std_pos_criterion,
        "std_all" : std_all_criterion,
        "min_max_neg" : min_max_neg_criterion,
        "min_max_pos" : min_max_pos_criterion,
        "min_max_all" : min_max_all_criterion,
        "mixed" : mixed_criterion,
    }
    
    if not criterion in criterion_map.keys():
        raise Exception(f"Threshold criterion {criterion} does not exists, pick one of {criterion_map.keys()}.")
    
    # return the thresholds based on the agressiveness
    return criterion_map[criterion]
    
    
def get_cluster_based_classification(samples_transformed : npt.ArrayLike,
                                     cluster_labels: npt.ArrayLike,
                                     axis_thresholds: npt.ArrayLike) -> npt.NDArray:
    """Generates an array with #threshold indicators per sample which indicates,
    whether the samples clusters mean is above the threshhold or not.

    All data is assumed to be in transformed coordinates, i.e. we
    assume the classification can be done simply per axis.

    This is an auxiliary function

    Args:
        samples_transformed (array_like): samples to classify
        cluster_labels (array_like): association of samples to clusters
        axis_thresholds (array_like): threshhold value per axis

    Returns:
        numpy ndarray : per sample indicator array indicating for every threshold,
            whether the sample is larger or smaller.
    """
    
    # get the clusters
    clusters = np.unique(cluster_labels)
    
    # count things
    num_thresholds = len(axis_thresholds)
    num_samples = len(cluster_labels)
    
    # for each sample we will store an indicator
    # array, whether its clusters center is above the
    # the threshold or not (for each threshold)
    sample_association = np.zeros((num_samples, num_thresholds))

    for cluster in clusters:
        
        # extract the samples
        cluster_mask = cluster_labels == cluster
        cluster_sample = samples_transformed[cluster_mask]
        
        # get sample mean
        cluster_mean = np.mean(cluster_sample, axis=0)
        
        for idx, threshold in enumerate(axis_thresholds):
            
            # decide if cluster is active in this dimension
            if cluster_mean[idx] > threshold:
                sample_association[cluster_mask, idx] = 1
            
    return sample_association

