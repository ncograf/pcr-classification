import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from typing import List
import transform_lib
from enum import Enum 

class ClusterHierarchyMeanClassifier(BaseEstimator):

    def __init__(self, 
                 negative_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 eps : float = 1.7,
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS']):
        """Initialize classifier with important parameters

            The algorithm is based on the three assumption:
            1. If there exists a cluster active in multiple dimensions, then in each of the active 
                dimensions there exists a cluster active only in the one dimension for each of the active dimensions
                (This was assured by the challange givers)
            2. The features are only positively coorelated
            3. Correlation cannot have a greater effect than acivity in one more dimension.
                This means if cluster_i is active in dimension 2 only and activity in 1 is correlated with
                activity in 2, then cluster_i cannot be higher in dimension 1 than a cluster which is
                in fact active in cluster 1.
            4. Clustering works good enough to recognize sensible clusters for in the best case it
                recognizes for each combination of labels exactly one cluster.


        Args:
            negative_control (npt.ArrayLike): Sample points used as negative control
            cluster_algorithm (ClusterMixin): Clustering algorithm to be used for builing the clusters
            eps (float, optional): Ratio threshhold for clusters to be considered larger. Defaults to 1.7.
            prediction_axis (List[str], optional): Assigns labels to the prediction axis. 
                Defaults to ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'].
        """
        
        # store local variables
        self.prediction_axis = prediction_axis
        self.cluster_algorithm = cluster_algorithm
        self.negative_control = negative_control
        self.negative_remover = transform_lib.RemoveNegativeTransformer(self.negative_control)
        self.eps = eps
        
        self.cluster_labels = None

    class Comparator(Enum):
        SMALLER = -1
        EQUAL = 0
        LARGER = 1
    
    def predict(self, X : npt.ArrayLike,
                y : npt.ArrayLike = None, 
                ) -> pd.DataFrame:
        """Assigns a list of true / false indicators for each input in X

        Args:
            X (npt.ArrayLike): Samples to be classified
            y (npt.ArrayLike, optional): Ignored. Defaults to None.

        Returns:
            pd.DataFrame: Indicators for each label
        """
        
        # only cluster positives
        self.X = X
        self.X_no_neg = self.negative_remover.transform(X)
        self.mask = self.negative_remover.mask
        
        self.clusters = self.cluster_algorithm.fit_predict(self.X_no_neg)
        self.cluster_list = self.split_clusters(data=self.X_no_neg, labels=self.clusters)

        self.outlier_mask = np.ones_like(self.mask, dtype=bool)
        self.outlier_mask[self.mask] = self.clusters >= 0
        
        self.min_dists = (np.max(self.X, axis=0) - np.min(self.X, axis=0)) * self.eps

        # make predictions
        self.cluster_predictions = self.compute_cluster_labels(cluster_list=self.cluster_list, min_dist=self.min_dists)
        self.no_neg_predictions = self.predict_clusters(self.cluster_predictions, self.clusters)
        
        # add all negatives
        all_predictions = np.zeros_like(X)
        all_predictions[self.mask] = self.no_neg_predictions
        
        # add labels to predictions
        self.predictions_df = pd.DataFrame(data = all_predictions, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def predict_clusters(self, cluster_predictions: npt.ArrayLike, cluster_labels: npt.ArrayLike ):
        predictions = np.zeros((cluster_labels.shape[0], cluster_predictions.shape[1]))
        for c in range(cluster_predictions.shape[0]):
            predictions[cluster_labels == c] = cluster_predictions[c]
        return predictions

    def split_clusters(self, data: np.ndarray, labels : np.ndarray) -> List[np.ndarray]:
        n_clusters = len(np.unique(labels)) - 1 # do not count outliers
        temp = [[] for _ in range(n_clusters)]

        # number of data points in total (number of rows)
        N = data.shape[0]

        for i in range(0, N):
            if labels[i] >= 0:
                temp[labels[i]].append(data[i, :])
            
        return [np.array(elem) for elem in temp]

    #returns feature coordinates mapped to the Comparator class based on which coordinates are different
    def compare_clusters(self, base_cluster : np.ndarray, other_cluster : np.ndarray, min_dist ) -> list[Comparator]:
        """ Based on the ratios base_mean / other_mean and other_mean / base_mean, using the threshhold
        the desition for base is Larger than other, base is Smaller than other or equal

        Args:
            base_cluster (np.ndarray): base cluster for comparison
            other_cluster (np.ndarray): other cluster for comparison
            eps (float, optional): Threshhold for ratio for base to be considered smaller or larger. Defaults to 1.7.

        Returns:
            list[Comparator]: Indication if base larger than other, base smaller than other or equal
        """
        
        # number of features
        dim = base_cluster.shape[1]
        assert dim == other_cluster.shape[1], "Number of features inconsistent!"

        # list of comparasings
        list = [self.Comparator.EQUAL]*dim

        # we are simply performing the comparison of the midpoints to determine whther the clusters are different
        base_mid = np.mean(base_cluster, axis=0)
        other_mid = np.mean(other_cluster, axis=0)
        for i in range(0, dim):
            if (base_mid[i] - other_mid[i] > min_dist[i]):
                list[i] = self.Comparator.LARGER
            elif (other_mid[i] - base_mid[i] > min_dist[i]):
                list[i] = self.Comparator.SMALLER
            else:
                list[i] = self.Comparator.EQUAL
        
        return list

    def compute_cluster_labels(self, cluster_list : list[np.ndarray], min_dist : float) -> np.ndarray:
        
        n_clusters = len(cluster_list)
        dim = cluster_list[0].shape[1]

        cluster_labels = np.zeros((n_clusters,dim))
        
        # A_{i,j,d} =  cluster_i <= cluster_j in dimension d
        A = np.ones((n_clusters,n_clusters,dim), dtype=bool)

        for j in range(n_clusters):
            for i in range(n_clusters):
                comparisons = self.compare_clusters(cluster_list[i], cluster_list[j], min_dist)
                for d in range(dim):
                    if comparisons[d] == self.Comparator.SMALLER:
                        A[i,j,d] = 1
                        A[j,i,d] = 0
                    elif comparisons[d] == self.Comparator.EQUAL:
                        A[i,j,d] = 1
                        A[j,i,d] = 1
                    elif comparisons[d] == self.Comparator.LARGER:
                        A[i,j,d] = 0
                        A[j,i,d] = 1
        
        # let cluster_i >=_d cluster_j denote the comparison of clusters i and j in dimension d
        # cluster_i is considered active in dimension d iff
        # \exists j : cluster_i >_d cluster_j and cluster_j >=_d' cluster_i \forall d' \neq d)
        for d in range(dim):
            for j in range(n_clusters):
                for i in range(n_clusters):
                    if A[i,j,d] == 0: # cluster_j <_d cluster_i
                        temp = 1
                        for d_ in range(dim):
                            if d_ == d: 
                                continue
                            if A[i,j,d_] == 0:
                                temp = 0
                        if temp == 1:
                            cluster_labels[i, d] = 1
        return cluster_labels