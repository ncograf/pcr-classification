
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from typing import List
import transform_lib
from icecream import ic

class ThresholdMeanClassifier(BaseEstimator):
    """Classifier, for the list of classes based on
    thresholds in transformed axis and clusterwise classifications
    
    The decorrelation only happens on the cluster means which
    eliminates the disbalances in point cardinality per cluster.
    
    Args:
        BaseEstimator : sklearn compatibility
    """
    
    def __init__(self, 
                 negative_control : npt.ArrayLike,
                 positive_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 transform_base : str = "pos",
                 whitening_transformer : TransformerMixin = transform_lib.WhitenTransformer(transform_lib.Whitenings.ZCA_COR),
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS']):
        """Initialize classifier with important parameters
        


        Args:
            negative_control (npt.ArrayLike): Sample points used as negative control
            positive_control (npt.ArrayLike): Sample points used as positive control
            cluster_algorithm (ClusterMixin): Clustering algorithm to be used for builing the clusters
            transform_base (str): cluster on which to compute the transform (negatives are always eliminated) e.g. "pos", "dynamic"
            whitening_transformer (TransformerMixin, optional): Whiteining transformer to be used
                in order to eliminate correlation. Defaults to transform_lib.WhitenTransformer(transform_lib.Whitenings.ZCA_COR).
            prediction_axis (List[str], optional): Assigns labels to the prediction axis. 
                Defaults to ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'].
        """
        
        # store local variables
        self.whitening_transformer = whitening_transformer
        self.prediction_axis = prediction_axis
        self.cluster_algorithm = cluster_algorithm
        self.negative_control = negative_control
        self.positive_control = positive_control
        self.transform_base = transform_base
        self.negative_remover = transform_lib.RemoveNegativeTransformer(self.negative_control)
        
        self.axis_threshholds = None
        self.X_transformed = None
        self.cluster_labels = None
        
    def fit(self, X : npt.ArrayLike, y : npt.ArrayLike = None):
        """Determine thresholds based on the input X
        
        Note that the input is ignored when the transform_base is equalt to "pos"
        
        Args:
            X (npt.ArrayLike): Sample
            y (npt.ArrayLike, optional): Ignored. Defaults to None.

        Returns:
            self
        """
        
        # transform base map
        transform_base_map = {
            "pos" : self.positive_control,
            "dynamic" : X
        }

        X_fit = transform_base_map[self.transform_base]
        X_no_neg = self.negative_remover.transform(X_fit)
        X_no_neg_scaled = X_no_neg / (np.max(X_no_neg, axis = 0) - np.min(X_no_neg, axis = 0))
        
        # generate clusters (under the assumption, that most points are negatives and get removed, this is fast)
        self.fit_clusters = self.cluster_algorithm.fit_predict(X_no_neg_scaled)

        # get cluster means
        self.fit_means = self.get_cluster_means(self.fit_clusters, X_no_neg)
        
        # create transformed inputs
        self.fit_means_whitened = self.whitening_transformer.fit_transform(self.fit_means)

        self.X_fit_transformed = self.whitening_transformer.transform(X_no_neg)
        self.negative_control_transformed = self.whitening_transformer.transform(self.negative_control)
        self.positive_control_transformed = self.whitening_transformer.transform(self.positive_control)
        
        # get threshholds
        self.axis_threshholds = self.get_per_axis_criterion()
        
        return self
        
    
    def predict(self, X : npt.ArrayLike,
                y : npt.ArrayLike = None, 
                ) -> pd.DataFrame:
        """Assigns a list of true / false indicators for each input in X
        
        The labels are determied based on the thresholds computed in fit (which has to be called
        before calling this). Moreover all points in the same cluster get the same labels, 
        according to the clusters mean.

        Args:
            X (npt.ArrayLike): Samples to be classified
            y (npt.ArrayLike, optional): Ignored. Defaults to None.

        Returns:
            pd.DataFrame: Indicators for each label
        """
        
        # only cluster positives
        self.X = X
        X_no_neg = self.negative_remover.transform(X)
        self.mask = self.negative_remover.mask
        X_no_neg_scaled = X_no_neg / (np.max(X_no_neg, axis = 0) - np.min(X_no_neg, axis = 0))
        
        # generate clusters (under the assumption, that most points are negatives and get removed, this is fast)
        self.clusters = self.cluster_algorithm.fit_predict(X_no_neg_scaled)

        # get cluster means
        self.means = self.get_cluster_means(self.clusters, X_no_neg)
        
        # create transformed inputs
        self.means_whitened = self.whitening_transformer.fit_transform(self.means)

        self.X_transformed = self.whitening_transformer.transform(X_no_neg)
        
        # store as well the transformed X for poltting
        self.X_all_transformed = self.whitening_transformer.transform(X)

        # make predictions
        predictions = get_cluster_based_classification(self.X_transformed, self.clusters, self.axis_threshholds)
        
        # add all negatives
        all_predictions = np.zeros_like(X)
        all_predictions[self.mask] = predictions
        
        # add labels to predictions
        self.predictions_df = pd.DataFrame(data = all_predictions, columns=self.prediction_axis)
    
        return self.predictions_df


    def get_cluster_means(self, clusters : npt.ArrayLike, data : npt.ArrayLike) -> npt.ArrayLike:
        """Generate an array of mean ponints for each cluster

        Args:
            clusters (array_like): cluster labels
            data (array_like): data to be clustered

        Returns:
            array_like: np array of means, one for each cluster
        """
        colors = np.unique(clusters)
        means = []
        for color in colors:
            X_color = data[clusters == color]
            means.append(np.mean(X_color, axis=0))
            
        return np.array(means)
        

    def get_per_axis_criterion(self) -> npt.NDArray:
        """Cluster based criterium for classification. The criterium
        will one threshold per axis in transformed coordinates.
        
        The idea is that in the transformed coordinates, there is no correlation between the axis
        and hence the individual classes / deseases should be separable along exactly one
        dimension via a threshold.

        For the determination of the threshhold, the same data as for the whitening is used.
        
        Returns:
            numpy ndarray : One threshold per input column as described above.
        """

        # compute min and max
        min_ = np.min(self.fit_means_whitened, axis=0)
        max_ = np.max(self.fit_means_whitened, axis=0)
        
        # min-max criterion
        min_max_criterion = (max_ + min_) / 2
        
        # return the thresholds based on the agressiveness
        return min_max_criterion
    
    
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

