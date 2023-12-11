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
from math import comb

class Comparator(Enum):
        MUCHSMALLER = -2
        SMALLER = -1
        EQUAL = 0
        LARGER = 1
        MUCHLARGER = 2

class ClusterRelativeHierarchyMeanClassifier(BaseEstimator):

    def __init__(self, 
                 cluster_algorithm : ClusterMixin,
                 whitening_transformer : TransformerMixin,
                 negative_control : npt.ArrayLike = None,
                 eps : float = 0.5,
                 contamination : float = 0.001,
                 negative_range: float = 0.9,
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'],
                 cutoff = 10000
                 ):
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
        self.cluster_labels = None
        self.whitening_transformer = whitening_transformer
        self.negative_control = negative_control
        self.negative_range=negative_range
        self.eps = eps
        self.outlier_quantile = contamination
        self.neg_dimensions = None
        self.thresh = 0.5
        self.probabilities_df = None
        self.X = None
        self.predictions_df = None
        self.point_hierarchy = None
        self.cutoff = self.cutoff

    def predict(self, X : npt.ArrayLike,
                y : npt.ArrayLike = None, 
                verbose : bool = False,
                ) -> pd.DataFrame:
        """Assigns a list of true / false indicators for each input in X

        Args:
            X (npt.ArrayLike): Samples to be classified
            y (npt.ArrayLike, optional): Ignored. Defaults to None.
            verbose (boo, optional): Defaults to None.

        Returns:
            pd.DataFrame: Indicators for each label
        """
        
        # ALL EXPOSED VARIABLES (CLASS PROPERTIES) MUST BE WITH RESPECT TO THE WHOLE SAMPLE
        # e.g. a mask must have length X.shape[0]
        
        self.read_data(data=X, negative_control=self.negative_control, negative_range=self.negative_range)
        
        self.predict_all()
    
        return self.predictions_df
    
    ###############################################
    # HIGH LEVEL METHODS
    ###############################################
    
    def predict_all(self):
        # make cluster predictions ==> use non-tranformed data
        self.cluster_dict = self.predict_cluster_labels(self.X_transformed,
                                                        clusters=self.cluster_dict,
                                                        eps=self.eps)

        # generate label predicitions
        self.predictions = self.predict_labels(clusters=self.cluster_dict, data=self.X)
        self.predictions[self.cluster_labels < 0,:] = -1
        self.probabilities = self.predictions
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = self.predictions, columns=self.prediction_axis)
        self.probabilities_df = pd.DataFrame(data = self.probabilities, columns=self.prediction_axis)
        
    
    def read_data(self, data : npt.NDArray, negative_control : npt.NDArray, negative_range : float):
        """Reads and clusters data makes all preparations for later changes

        Args:
            data (npt.NDArray): data to be processed
            negative_control (npt.NDArray): negative contorl sample
            negative_range (float) : range of the negative controls max in each dimension to be considerd negative
        """
        # set data
        self.X = data

        # set negatibe control
        self.negative_remover = transform_lib.RemoveNegativeTransformer(negative_control, range=negative_range)
        self.negative_control = negative_control
        _ = self.negative_remover.transform(self.X)
        self.No_neg_mask = self.negative_remover.mask


        # detect outliers
        outlier_detector = IsolationForest(contamination=self.outlier_quantile,
                                            n_jobs=3,
                                            max_samples=self.X.shape[0],
                                            n_estimators=10)
        outliers_labels = outlier_detector.fit_predict(self.X)
        outliers_mask = outliers_labels < 0

        # compute zero dimensions
        self.neg_dimensions = np.percentile(self.X[~outliers_mask],99.99999,axis=0) <= self.cutoff
        
        # remove outliers and create clusters
        self.cluster_labels = self.get_clusters(self.X, self.cluster_algorithm, outliers_mask=outliers_mask, no_neg_mask=self.No_neg_mask) 
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
    
   ################################################
   # LOW LEVEL METHODS
   ################################################ 
        
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

    def add_transformed(self, data_t: npt.ArrayLike, clusters : Dict[int,transform_lib.Cluster], percentile = 0.01) -> Dict[int, transform_lib.Cluster]:
        percentile = percentile * 100
        for k in clusters.keys():
            clusters[k].mean_t = np.mean(data_t[clusters[k].mask,:], axis=0)
            clusters[k].max_t = np.max(data_t[clusters[k].mask,:], axis=0)
            clusters[k].min_t = np.min(data_t[clusters[k].mask,:], axis=0)
            clusters[k].low_perc_t = np.percentile(data_t[clusters[k].mask,:], axis=0, q=percentile)
            clusters[k].high_perc_t = np.percentile(data_t[clusters[k].mask,:], axis=0, q=100-percentile)
        
        return clusters
    
    def get_clusters(self, data : npt.NDArray, cluster_engine : ClusterMixin, outliers_mask : npt.NDArray, no_neg_mask : npt.NDArray) -> npt.NDArray:
        labels = np.ones(data.shape[0], dtype=int) * (-1)
        labels[~outliers_mask & no_neg_mask] = cluster_engine.fit_predict(data[~outliers_mask & no_neg_mask])
        labels[np.logical_not(no_neg_mask)] = np.max(labels) + 1 # negative control range is last cluster
        return labels
    
    def select_max_dimensions(self, clusters : Dict[int, transform_lib.Cluster],
                              num_active : np.ndarray,
                              dimensions : np.ndarray,
                              dim : int) -> List[np.ndarray]:
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
                # exclude unused dimensions
                temp = np.multiply(clusters[i].mean, np.invert(self.neg_dimensions))

                # scale the dimensions and find the largest ones
                temp = np.argsort(-np.divide(temp, dimensions))[:num_active[i]]

                # mark the num_active[i] highest dimensions as active
                for j in temp:
                    if clusters[i].mean[j] > 0.7 * self.cutoff:
                        active[i][j] = True

        return active

    def compute_hierarchy(self, comparators : np.ndarray,  n_clusters : int, dim : int, new_dim : int) -> Dict[int, transform_lib.Cluster]:
    
        # consider all pairs of clusters
        pairs = list(itertools.combinations(range(n_clusters), 2))
        hierarchy = np.zeros(n_clusters, int)
    
        # we define rank to refer to the number of active features
        for rank in range(new_dim):
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
                if (Comparator.MUCHSMALLER in outcome) and (Comparator.LARGER not in outcome) and (Comparator.MUCHLARGER not in outcome):
                    base_superior = True
    
                # other is hence active in at least = rank coordinates, hence deserves a higher rank
                if other_superior:
                    hierarchy[other] = rank + 1
    
                if base_superior:
                    hierarchy[base] = rank + 1

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
        comparators = np.ndarray((n_clusters, n_clusters, dim), dtype=Comparator)
        comparators.fill(Comparator.EQUAL)

        for i in range(n_clusters):
            for j in range(n_clusters):
                for k in range(dim):

                    # skip inactive dimension
                    if self.neg_dimensions[k]:
                        continue

                    # compare cluster i against cluster j in every dimension
                    if ((clusters[i].mean[k] - clusters[j].mean[k]) > (eps)*dimensions[k]):
                        comparators[i][j][k] = Comparator.MUCHLARGER

                    elif ((clusters[i].mean[k] - clusters[j].mean[k]) > (eps / 10.0) * dimensions[k]):
                        comparators[i][j][k] = Comparator.LARGER

                    elif ((clusters[j].mean[k] - clusters[i].mean[k]) > (eps) * dimensions[k]):
                        comparators[i][j][k] = Comparator.MUCHSMALLER
                    
                    elif ((clusters[j].mean[k] - clusters[i].mean[k]) > (eps / 10.0) * dimensions[k]):
                        comparators[i][j][k] = Comparator.SMALLER
                    
                    else:
                        comparators[i][j][k] = Comparator.EQUAL

    
        return comparators
    
    def predict_cluster_labels(self, 
                               data : npt.ArrayLike,
                               clusters : Dict[int, transform_lib.Cluster],
                               eps : float) -> Dict[int, transform_lib.Cluster]:
        """Predict labels for eac cluster and store them in the cluter dictionary

        Args:
            data (array_like): All data points
            clusters (Dict[int, Cluters]): Dict containing the clusters
            eps (float): factor for the minimal disance relative to the max cluter means

        Returns:
            np.ndarray: Indicators for diseases present in each cluster
        """
        clusters_tmp = clusters.copy()
        
        clusters_tmp.pop(-1, None) # get rid of outliers

        n_clusters = len(clusters_tmp)
        
        if (n_clusters == 0):
            return clusters
        
        dim = data.shape[1]
        N = data.shape[0]
        # only relevant number of dimensions
        new_dim = dim - np.count_nonzero(self.neg_dimensions)

        # compute axis scaling for each dimension, without considering outliers
        maximums = clusters_tmp[0].mean
        minimums = clusters_tmp[0].mean

        for i in range(n_clusters):
            minimums = np.minimum(clusters_tmp[i].mean, minimums)
            maximums = np.maximum(clusters_tmp[i].mean, maximums)

        dimensions = maximums - minimums

        # default eps was not provided
        if eps == None:
            # use the method of closest fit to binomial to find an epsilon producing the best hierarchy
            eps_min = 0.3
            eps_max = 0.8
            steps = 16
            step = (eps_max - eps_min) / steps

            # make a shifted binomial (to account for dillution and normal clusters)
            goal =  np.array([comb(new_dim, i) for i in range(new_dim+1)] + [0]*(dim - new_dim))

            all_errors = np.ndarray(steps, dtype = float)
            eps_proposal = eps_min

            for i in range(steps):
                comparators = self.compute_comparators(clusters_tmp, dimensions, eps_proposal, dim)
                hierarchy = self.compute_hierarchy(comparators, n_clusters, dim, new_dim)

                # compute the binomail fit of the hierarchy
                counts = np.bincount(hierarchy, minlength=(dim+1))
                # scale them to the perfect number of clusters
                counts = counts * min(1, 2**new_dim / n_clusters) 
                # compute the L1 norm for the difference
                error = np.linalg.norm(goal-counts, ord=1)

                all_errors[i] = error
                eps_proposal += step

            # select best epsilon
            eps = eps_min + step * (np.argmin(all_errors))

        print(f"Epsilon: {eps}")

        # select the dimensions where the clsuter is most active
        comparators = self.compute_comparators(clusters_tmp, dimensions, eps, dim)
        hierarchy = self.compute_hierarchy(comparators, n_clusters, dim)
        labels = self.select_max_dimensions(clusters_tmp, hierarchy, dimensions, dim)

        # store point hierarchy for debug
        point_hierarchy = np.full(N, -1)
        for i in range(N):
            point_hierarchy[i] = hierarchy[self.cluster_labels[i]]

        self.point_hierarchy = point_hierarchy

        print(f"Hierarchy: {hierarchy}")

        # assign the labels to the clusters
        for i in range(n_clusters):
            clusters[i].labels = labels[i]
        
        return clusters