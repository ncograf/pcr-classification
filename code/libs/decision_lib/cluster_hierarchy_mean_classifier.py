import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import transform_lib
from enum import Enum 

class ClusterHierarchyMeanClassifier(BaseEstimator):

    def __init__(self, 
                 negative_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 whitening_transformer : TransformerMixin,
                 eps : float = 1.7,
                 contamination : float = 0.001,
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
        self.whitening_transformer = whitening_transformer
        self.negative_control = negative_control
        self.negative_remover = transform_lib.RemoveNegativeTransformer(self.negative_control)
        self.eps = eps
        self.contamination = 0.001
        
        self.cluster_labels = None

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
        self.X_no_neg_mask = self.negative_remover.mask
        
        self.clusters_no_neg = self.get_clusters_and_outlier(self.X_no_neg, self.cluster_algorithm, contamination=self.contamination)
        self.cluster_no_neg_masks = self.split_clusters(labels=self.clusters_no_neg)
        
        # mark outliers using boolean masks
        self.outlier_no_neg_mask = self.clusters_no_neg < 0
        self.outlier_all_mask = np.zeros_like(self.X_no_neg_mask, dtype=bool)
        self.outlier_all_mask[self.X_no_neg_mask] = self.outlier_no_neg_mask
        
        # The removed zero cluster is added as a last cluster
        self.clusters_all = np.zeros_like(self.X_no_neg_mask)
        self.clusters_all[self.X_no_neg_mask] = self.clusters_no_neg
        self.clusters_all[np.logical_not(self.X_no_neg_mask)] = np.max(self.clusters_no_neg) + 1
        
        # get cluster means
        self.cluster_means = []
        for k in self.cluster_no_neg_masks.keys():
            if not k == -1:
                mask = self.cluster_no_neg_masks[k]
                self.cluster_means.append(np.mean(self.X_no_neg[mask], axis = 0))
        self.cluster_means = np.stack(self.cluster_means, axis=0)
        
        # transform clusters
        self.whitening_transformer.fit(self.cluster_means)
        self.X_no_neg_transformed = self.whitening_transformer.transform(self.X_no_neg)
        self.df_X_all_transformed = pd.DataFrame(data=self.whitening_transformer.transform(self.X), columns=self.prediction_axis)
        
        # make predictions
        self.cluster_predictions_no_neg = self.compute_cluster_labels(self.X_no_neg_transformed, cluster_mask=self.cluster_no_neg_masks, eps=self.eps)
        self.predictions_no_neg = self.predict_clusters(self.cluster_predictions_no_neg, self.clusters_no_neg, self.cluster_no_neg_masks)
        self.predictions_no_neg[self.outlier_no_neg_mask,:] = -1
        
        # add all negatives
        all_predictions = np.zeros_like(X)
        all_predictions[self.X_no_neg_mask] = self.predictions_no_neg
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = all_predictions, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def predict_clusters(self,
                         cluster_predictions: Dict[int, npt.ArrayLike],
                         cluster_labels: npt.ArrayLike,
                         cluster_masks: Dict[int, npt.ArrayLike]) -> npt.NDArray:
        """Get from cluster prediction to point predictions using the cluster assignments

        Args:
            cluster_predictions (Dict[int, array_like]]): Predictions of diseases for each cluster
            cluster_labels (npt.ArrayLike): Points to cluster mapping
            cluster_mask (Dict[int, array_like]): masks for each cluster

        Returns:
            NDArray: Predictions on points
        """
        predictions = np.zeros((cluster_labels.shape[0], cluster_predictions[0].shape[0]))
        for c in cluster_masks.keys():
            predictions[cluster_masks[c]] = cluster_predictions[c]
        return predictions

    def split_clusters(self, labels : np.ndarray) -> Dict[int, npt.NDArray]:
        """Get list, containing masks for clusters

        Args:
            labels (np.ndarray): cluster assignments

        Returns:
            Dict[int, np.ndarray]: list containing the cluster masks
        """
        cluster_masks = {}
        
        for label in np.unique(labels):
            cluster_masks[label] = labels == label

        return cluster_masks

    #returns feature coordinates mapped to the Comparator class based on which coordinates are different
    def compare_clusters(self, base_cluster : np.ndarray, other_cluster : np.ndarray) -> npt.NDArray:
        """ Based on the ratios base_mean / other_mean and other_mean / base_mean, using the threshhold
        the desition for base is Larger than other, base is Smaller than other or equal

        Args:
            base_cluster (np.ndarray): base cluster for comparison
            other_cluster (np.ndarray): other cluster for comparison

        Returns:
            NDArray: Indication if base larger than other, base smaller than other or equal
        """
        
        # number of features
        dim = base_cluster.shape[1]
        assert dim == other_cluster.shape[1], "Number of features inconsistent!"

        # we are simply performing the comparison of the midpoints to determine whther the clusters are different
        base_mid = np.mean(base_cluster, axis=0)
        other_mid = np.mean(other_cluster, axis=0)
        return base_mid - other_mid
    
    def get_clusters_and_outlier(self, data : npt.ArrayLike, cluster_engine : ClusterMixin, get_outliers=True, contamination = 0.0005) -> npt.NDArray:

        if get_outliers:
            outlier_detector = IsolationForest(contamination=contamination,
                                               n_jobs=4,
                                               max_samples=data.shape[0],
                                               n_estimators=10)
            labels = outlier_detector.fit_predict(data) # outliers will get label -1
            labels[labels >= 0] = cluster_engine.fit_predict(data[labels >= 0])
        else:
            labels = cluster_engine.fit_predict(data)
        
        return labels



    def compute_cluster_labels(self,data : npt.ArrayLike, cluster_mask : List[npt.ArrayLike], eps : float) -> npt.NDArray:
        """Predict labels for each cluster

        Args:
            data (array_like): All data points
            cluster_list (List[array_like]): List containing the clusters
            eps (float): factor for the minimal disance relative to the max

        Returns:
            np.ndarray: Indicators for diseases present in each cluster
        """
        
        if -1 in cluster_mask.keys(): 
            cluster_mask.pop(-1) # get rid of outliers
        n_clusters = len(cluster_mask)
        dim = data.shape[1]

        cluster_labels = dict([(i,np.zeros(dim)) for i in cluster_mask.keys()])
        
        # A_{i,j,d} =  cluster_i - cluster_j in dimension d (where we compare midpoints)
        A = np.ones((n_clusters,n_clusters,dim), dtype=float)

        for j in cluster_mask.keys():
            for i in cluster_mask.keys():
                dists = np.mean(data[cluster_mask[j]],axis=0) - np.mean(data[cluster_mask[i]], axis = 0)
                A[j,i] = dists
        
        D = np.max(A, axis=(0,1)) - np.min(A, axis=(0,1))
        
        # let A_jid denote the directed distance cluster_j - cluster_i between midpoints in dimension d
        # let D_d denote the maximal distance between two clusters
        # cluster_i is considered active in dimension d iff
        # \exists j : A_ijd > eps * D_d and A_ijd < -eps * D_d' \forall d' \neq d)
        # in words this requres c_i to be lareger in dimension c_j by at least some sparating distace
        #       and it requires c_i to be within the range of c_j for the other dimensios
        for d in range(dim):
            for j in cluster_mask.keys():
                for i in cluster_mask.keys():
                    if A[i,j,d] > eps * D[d]: # cluster_j <_d cluster_i
                        temp = 1
                        for d_ in range(dim):
                            if d_ == d: 
                                continue
                            if A[i,j,d_] < -1.2 * eps * D[d_]: # cluster_j >=_d' cluster_i
                                temp = 0
                        if temp == 1:
                            cluster_labels[i][d] = 1
        return cluster_labels