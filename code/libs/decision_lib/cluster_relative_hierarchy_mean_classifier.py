import numpy as np
import pandas as pd
import scipy.stats as sp
import numpy.typing as npt
import transform_lib
import itertools
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from typing import Dict, List
from icecream import ic
from enum import Enum

class Comparator(Enum):
        MUCHSMALLER = -2
        SMALLER = -1
        EQUAL = 0
        LARGER = 1
        MUCHLARGER = 2

class ClusterRelativeHierarchyMeanClassifier(BaseEstimator):

    def __init__(self, 
                 negative_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 whitening_transformer : TransformerMixin,
                 eps : float = 1.7,
                 contamination : float = 0.001,
                 negative_range: float = 0.9,
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS']):
        """Initialize classifier with important parameters

            The algorithm is based on the three assumption:
            1. If there exists a cluster K of which the points are active k dimension, 
                then for each dimension d in which K is active, there exists cluster K_d which 
                active in the same dimensions as K except for dimension d.
                (Needed for decision based on negative cluster in same dimension)
            2. For each dimension, there exists positive clusters.
                (This can be ensured with a positive control sample, used for decorrelation.)
            4. Clustering works good enough to recognize sensible clusters.
                (This can be ensured by taking enough clusters)

        Args:
            negative_control (npt.ArrayLike): Sample points used as negative control
            cluster_algorithm (ClusterMixin): Clustering algorithm to be used for builing the clusters
            whitening_transformer (TransformerMixin): Whitening algorithm to decorrelate clusters
            eps (float, optional): Ratio threshhold for clusters to be considered larger. Defaults to 1.7.
            contaminaton (float, optional): Ratio of outliers in the data. Defaults to 0.001.
            negative_range (float, optional): The range of datapoints (relative to max of negative control) to be considered negative without further computations
            prediction_axis (List[str], optional): Assigns labels to the prediction axis. Defaults to ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'].
        """
        
        # store local variables
        self.prediction_axis = prediction_axis
        self.cluster_algorithm = cluster_algorithm
        self.whitening_transformer = whitening_transformer
        self.negative_control = negative_control
        self.negative_remover = transform_lib.RemoveNegativeTransformer(self.negative_control, range=negative_range)
        self.eps = eps
        self.contamination = contamination
        
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
        
        # ALL EXPOSED VARIABLES (CLASS PROPERTIES) MUST BE WITH RESPECT TO THE WHOLE SAMPLE
        # e.g. a mask must have lenth X.shape[0]
        
        # remove points in negative control range
        self.X = X
        X_no_neg = self.negative_remover.transform(X)
        self.No_neg_mask = self.negative_remover.mask
        
        # remove outliers and create clusters
        cluster_labels_no_neg = self.get_clusters_and_outlier(X_no_neg, self.cluster_algorithm, contamination=self.contamination)
        self.cluster_labels = np.zeros_like(self.No_neg_mask, dtype=int)
        self.cluster_labels[self.No_neg_mask] = cluster_labels_no_neg
        self.cluster_labels[np.logical_not(self.No_neg_mask)] = np.max(cluster_labels_no_neg) + 1 # negative control range is last cluster
        self.cluster_dict : Dict[int, transform_lib.Cluster] = self.split_clusters(data=self.X,labels=self.cluster_labels)
        
        # transform data accoring to clusters
        cluster_dict_no_outlier = list(self.cluster_dict.keys())
        if -1 in cluster_dict_no_outlier:
            cluster_dict_no_outlier.remove(-1)
        cluster_means_no_outlier = np.stack([self.cluster_dict[k].mean for k in cluster_dict_no_outlier], axis=0)
        self.whitening_transformer.fit(cluster_means_no_outlier)
        self.X_transformed = self.whitening_transformer.transform(self.X)
        
        # add transformed properties to clusters
        self.cluster_dict = self.add_transformed(self.X_transformed, self.cluster_dict)
        
        # make cluster predictions ==> use non-tranformed data
        self.cluster_dict = self.predict_cluster_labels(self.X, clusters=self.cluster_dict, eps=self.eps)

        # generate label predicitions
        self.predictions = self.predict_labels(clusters=self.cluster_dict, data=self.X)
        self.predictions[self.cluster_labels < 0,:] = -1
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = self.predictions, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def predict_labels(self,
                        clusters : Dict[int, transform_lib.Cluster],
                        data : npt.ArrayLike) -> npt.NDArray:
        """Get from cluster prediction to point predictions using the cluster assignments

        Args:
            clusters (Dict[int, Cluster]]): cluster dictonary containing labels and cluster masks
            data (npt.ArrayLike): Data points used for data shape

        Returns:
            NDArray: Predictions on points
        """
        predictions = np.zeros_like(data)
        for c in clusters.keys():
            predictions[clusters[c].mask] = clusters[c].labels
        return predictions

    def split_clusters(self, data : npt.ArrayLike, labels : npt.ArrayLike) -> Dict[int, transform_lib.Cluster]:
        """Get dictonariy of clusters

        Args:
            data (array_like): all data_points
            labels (array_like): cluster assignments

        Returns:
            Dict[int, ndarray]: dictonary containing the clusters
        """
        clusters = {}
        
        for label in np.unique(labels):
            clusters[label] = transform_lib.Cluster(data, labels == label)

        return clusters

    def add_transformed(self, data_t: npt.ArrayLike, clusters : Dict[int,transform_lib.Cluster]) -> Dict[int, transform_lib.Cluster]:
        for k in clusters.keys():
            clusters[k].mean = np.mean(data_t[clusters[k].mask,:], axis=0)
            clusters[k].max = np.max(data_t[clusters[k].mask,:], axis=0)
            clusters[k].min = np.min(data_t[clusters[k].mask,:], axis=0)
        
        return clusters
    
    def get_clusters_and_outlier(self, data : npt.ArrayLike, cluster_engine : ClusterMixin, get_outliers=True, contamination = 0.001) -> npt.NDArray:

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
    
    def select_max_dimensions(self, clusters : Dict[int, transform_lib.Cluster], num_active : np.ndarray, dimensions : np.ndarray, dim : int) -> np.ndarray:
        """
        Given that we know the number of dimensions the clusters are active to classify them we only select the number of dimensions
        they are active in

        Args:
            clusters (Dict[int, Cluters]): Dict containing the clusters
            num_active (float): classification accroding to number fo active labels
            dimensions (float): min-max for each dimension across all clusters
            dim : dimensionality of clusters        
        """

        assert len(clusters) == len(num_active), "Inconsistent number of labellings for clusters"
        assert len(dimensions) == dim, "Inconsistent number of dimensions"
        assert np.all(num_active >= 0), "Inconsistent number of active labels provided"
        assert np.all(num_active <= dim), "Inconsistent number of active labels provided"

        n_clusters = len(clusters)
        active = np.zeros((n_clusters, dim), dtype=bool)

        for i in range(n_clusters):

            if (num_active[i] > 0):
                # scale the dimensions and find the largest ones
                temp = np.argsort(-np.divide(clusters[i].mean, dimensions))[:num_active[i]]

                for j in temp:
                    active[i][j] = True

        return active


    
    def compute_hierarchy(self, comparators : np.ndarray,  n_clusters : int, dim : int) -> Dict[int, transform_lib.Cluster]:
    
        # consider all pairs of clusters
        pairs = list(itertools.combinations(range(n_clusters), 2))
        hierarchy = np.zeros(n_clusters, int)
    
        # we define rank to refer to the number of active features
        for rank in range(dim):
            for (base, other) in pairs:
                # we only consider clusters of given rank
                if hierarchy[base] < rank or hierarchy[other] < rank:
                   continue
                
                outcome = comparators[other, base, : ]
    
                # decide whether the other_cluster is strictly more active then base
                other_superior : bool = False
                base_superior : bool = False
    
                # more active = more active in one coordinate
                if (Comparator.MUCHLARGER in outcome) and (Comparator.SMALLER not in outcome) and (Comparator.MUCHSMALLER not in outcome):
                    other_superior = True
    
                # more active = at least as active in other coordinates
                if (Comparator.MUCHLARGER not in outcome) and (Comparator.LARGER not in outcome) and (Comparator.MUCHSMALLER in outcome):
                    base_superior = True
    
                # other is hence active in at least = rank coordinates, hence deserves a higher rank
                if other_superior:
                    hierarchy[other] = rank + 1
    
                if base_superior:
                    hierarchy[base] = rank + 1

        print(hierarchy)
    
        return hierarchy
    
    
    def compute_comparators(self, clusters : Dict[int, transform_lib.Cluster], dimensions, eps : float, dim : int) -> np.ndarray:
        """ Compare all clusters in all dimensions and return a matrix

        Args:
            clusters (Dict[int, Cluters]): Dict containing the clusters
            eps (float): factor for the minimal disance relative to the max cluter means
            dim : dimensionality of clusters

        Returns:
            all clusters compared with all clusters in all dimensions as a 3d np.ndarray
        """

        n_clusters = len(clusters)
        comparators = np.zeros((n_clusters, n_clusters, dim), dtype=Comparator)

        for i in range(n_clusters):
            for j in range(n_clusters):
                for k in range(dim):

                    # compare cluster i against cluster j in every dimension
                    if ((clusters[i].mean[k] - clusters[j].mean[k]) > eps * dimensions[k]):
                        comparators[i][j][k] = Comparator.MUCHLARGER

                    elif ((clusters[i].mean[k] - clusters[j].mean[k]) > 0.5 * eps * dimensions[k]):
                        comparators[i][j][k] = Comparator.LARGER

                    elif ((clusters[j].mean[k] - clusters[i].mean[k]) > eps * dimensions[k]):
                        comparators[i][j][k] = Comparator.MUCHSMALLER
                    
                    elif ((clusters[j].mean[k] - clusters[i].mean[k]) > 0.5 * eps * dimensions[k]):
                        comparators[i][j][k] = Comparator.SMALLER
                    
                    else:
                        comparators[i][j][k] = Comparator.EQUAL

    
        return comparators
    
    def predict_cluster_labels(self,data : npt.ArrayLike, clusters : Dict[int, transform_lib.Cluster], eps : float) -> Dict[int, transform_lib.Cluster]:
        """Predict labels for eac cluster and store them in the cluter dictionary

        Args:
            data (array_like): All data points
            clusters (Dict[int, Clusters]): Dict containing the clusters
            eps (float): factor for the minimal disance relative to the max cluter means

        Returns:
            np.ndarray: Indicators for diseases present in each cluster
        """
        
        clusters_tmp = clusters.copy()
        
        clusters_tmp.pop(-1, None) # get rid of outliers
        n_clusters = len(clusters_tmp)
        dim = data.shape[1]

        # compute axis scaling for each dimension, without considering outliers
        temp = np.ndarray((dim, 2), dtype=float)
        for i in range(n_clusters):
            for k in range(dim):
                temp[:, 0] = np.minimum(clusters_tmp[i].mean, temp[:, 0])
                temp[:, 1] = np.maximum(clusters_tmp[i].mean, temp[:, 1])

        dimensions = np.ndarray(dim, dtype=float)
        dimensions = temp[:, 1]- temp[:, 0]

        comparators = self.compute_comparators(clusters_tmp, dimensions, eps, dim)
        hierarchy = self.compute_hierarchy(comparators, n_clusters, dim)
        labels = self.select_max_dimensions(clusters_tmp, hierarchy, dimensions, dim)

        for i in range(n_clusters):
            clusters[i].labels = labels[i]
        
        return clusters