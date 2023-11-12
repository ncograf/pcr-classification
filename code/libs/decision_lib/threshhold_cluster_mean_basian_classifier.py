import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from typing import List
from scipy.stats import multivariate_normal
import transform_lib

class ThresholdMeanBayesianClassifier(BaseEstimator):
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
        predictions = self.get_cluster_based_basian_classification(self.X_transformed, self.clusters, self.axis_threshholds)
        
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
    
    def get_cluster_based_basian_classification(
                                        self,
                                        samples_transformed : npt.ArrayLike,
                                        cluster_labels: npt.ArrayLike,
                                        axis_thresholds: npt.ArrayLike) -> npt.NDArray:
        r"""Computes for each point a vector of probabilies, whether the point
        is infected with the corresponding disease.
        
        We assume that each clusters have a probability that it is positive
        for some disease. This probabiltiy can be determined in variaous way
        - 1 or 0 -> hard threshold
        - assumption of two gaussians computed based on the clusters or points
        - any other distribution assumption

        Then we compute for each point the probability of the point having a disease
        based on the assumption that every point in a cluster has the same disease.
        So we need to estimate the probability that a point belongs to a given cluster.

        The probability is computed as follows:
        
        :math:`Z_i := indicator for a point having disease i.`

        :math:`C := the cluster the point belongs to.`

        :math:`K_k := indicator for cluter k to have disease .`
        
        :math:`P[Z_i = 1] = \sum_k P[Z_i = 1 | C = k] P[C = k] = \sum_k P[K_k = 1] P[C = k]`
        
        All in all, we make the following assumptions:
        
        1. All data is assumed to be in transformed coordinates, i.e. we assume the classification can be done simply per axis.
        2. All the points in one cluster have a disease or none of the points hase a disease
        3. The distribution conditional on the clusters mean, if a cluster 
            has a disease or not: :math:`P[K_k = 1]`
        4. The distribution of the point conditional on the position, if 
            it belongs to a cluster: :math:`P[C = k]`
            
        The thrid assumption will be a hard threshold at first.
        The forth assumption we assume each cluster to have a gaussian distribution,
            then we compute a softmax over the points.

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
        
        # store the cluster probabilies for each dimension
        cluster_probabilites = []
        
        # allocate list for pdf values
        sample_pdfs = []
        
        for cluster in clusters:
            
            # extract the samples
            cluster_mask = cluster_labels == cluster
            cluster_sample = samples_transformed[cluster_mask]
            
            # get sample mean and cov
            cluster_mean = np.mean(cluster_sample, axis=0)
            cluster_cov = np.cov(cluster_sample.T)

            local_probs = np.zeros(len(axis_thresholds))
            for idx, threshold in enumerate(axis_thresholds):
                
                # decide if cluster is active in this dimension
                if cluster_mean[idx] > threshold:
                    local_probs[idx] = 1
        
            # add cluster disease probs for each channel and the local culster to the list
            cluster_probabilites.append(local_probs)

            # if the cluster is too small for any computation go to the next
            if cluster_sample.shape[0] <= samples_transformed.shape[1]:
                sample_pdfs.append(np.zeros(samples_transformed.shape[0]))
                continue
            
            # probability desities at the points
            pdf = multivariate_normal.pdf(samples_transformed, 
                                    mean=cluster_mean,
                                    cov=cluster_cov,
                                    allow_singular=False) 

            sample_pdfs.append(pdf)

        # (n_clusters, n_threshholds)
        cluster_probabilites = np.stack(cluster_probabilites, axis=0)
        
        # (n_samples, n_clusters): prob for each point to belong to a given cluster
        sample_pdfs = np.stack(sample_pdfs, axis=1)
        
        # (n_samples)
        sample_pdf_sum = np.sum(sample_pdfs, axis=1)
        
        # add up the individual cluster probabilies
        eps = 1e-20
        p_c_eq_k = np.einsum("jk,ki->ji", sample_pdfs, cluster_probabilites)
        p_c_eq_k = np.einsum("jk,j->jk", p_c_eq_k, 1 / (sample_pdf_sum + eps))
        p_c_eq_k = np.clip(p_c_eq_k, a_min = 0, a_max=1)

        return p_c_eq_k

