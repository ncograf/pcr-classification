import numpy as np
import itertools
import pandas as pd
import numpy.typing as npt
import transform_lib
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Callable, ParamSpec, TypeVar
from negative_dimnesion_detection import get_negative_dimensions
from icecream import ic
import time

def print_time(text : str):
    """Decorator to print time necessary for the computation
    of the function. The time will be printed along the
    given `text`

    Args:
        text (str): Text to be printed with time
    """
    T = TypeVar("T")
    P = ParamSpec("P")
    def decorator(f : Callable[P, T]) -> Callable[P, T]:
        def _inner(self, *args: P.args, **kwargs: P.kwargs) -> T:
            start = time.time()
            
            # execute acual function call
            result = f(self, *args, **kwargs)

            elapsed = time.time() - start
            elapsed = elapsed
            if self.verbose:
                print(f'Finished {text} in {elapsed} seconds')
            return result
        return _inner
    return decorator

class ZeroDensity():
    def score_samples(self, X):
        return np.log(np.ones(X.shape[0]) * 1e-15)

class WhitnesDensityClassifier(BaseEstimator):

    def __init__(self, 
                 cluster_algorithm : ClusterMixin,
                 whitening_transformer : TransformerMixin = transform_lib.WhitenTransformer(transform_lib.Whitenings.NONE),
                 negative_control : npt.ArrayLike = None,
                 eps : float = 1.7,
                 outlier_quantile: float = 0.001,
                 negative_range: float = 0.9,
                 prediction_axis : List[str] = ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'],
                 verbose = False):
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
            outlier (bool, optional): Flag indicating if outlier should be detected, note that some cluster algorithms do this automatically. Default True.
            desnity_quantile (float, optional): Quantile in [0,1] of positive / negative points for which to compute densities. 
                100-density_quantile will percent will be assigned probaiblity 1 or 0 based on the precomputed labels.
            prediction_axis (List[str], optional): Assigns labels to the prediction axis. Defaults to ['SARS-N2_POS','SARS-N1_POS','IBV-M_POS','RSV-N_POS','IAV-M_POS','MHV_POS'].
        """
        
        # store local variables
        self.prediction_axis = prediction_axis
        self.cluster_algorithm = cluster_algorithm
        self.whitening_transformer = whitening_transformer
        self.negative_control = negative_control
        self.negative_remover = transform_lib.RemoveNegativeTransformer(self.negative_control, range=negative_range)
        self.negative_range=negative_range
        self.eps = eps
        self.outlier_quantile = outlier_quantile
        self.neg_dimensions = None
        self.verbose = verbose
        self.thresh = 0.5
        self.probabilities_df = None
        self.X = None
        self.predictions_df = None

        
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
    # HIGH LEVEL AND STATIC METHODS
    ###############################################
    
    def predict_all(self):
        # make cluster predictions
        self.cluster_dict = self.predict_cluster_labels(self.X,
                                                        clusters=self.cluster_dict,
                                                        eps=self.eps,
                                                        zero_dimensions=self.neg_dimensions)

        # generate label predicitions
        self.probabilities, self.predictions = self.predict_labels(clusters=self.cluster_dict, data=self.X, thresh=self.thresh)
        self.predictions[self.cluster_labels < 0,:] = -1
        self.probabilities[self.cluster_labels < 0,:] = -1
        
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
        self.neg_dimensions = np.percentile(self.X[outliers_mask],99,axis=0) <= 10000
        
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
    
    @print_time("predict labels")
    def predict_labels(self,
                        clusters : Dict[int, transform_lib.Cluster],
                        data : npt.ArrayLike,
                        thresh : float) -> npt.NDArray:
        """Get from cluster prediction to point predictions using the cluster assignments

        Args:
            clusters (Dict[int, Cluster]]): cluster dictonary containing labels and cluster masks
            data (npt.ArrayLike): Data points used for data shape
            thresh (float): Threshold for data to be considered active or not

        Returns:
            NDArray: Predicted probabilies on points
        """
        predictions = np.zeros_like(data)
        probabilies = np.zeros_like(data, dtype=np.float32)
        for c in clusters.keys():
            probabilies[clusters[c].mask] = clusters[c].active_probs
            predictions[clusters[c].mask] = clusters[c].active_probs >= thresh

        return probabilies, predictions

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

    @print_time("compute transformed features")
    def add_transformed(self, data_t: npt.ArrayLike, clusters : Dict[int,transform_lib.Cluster], percentile = 0.01) -> Dict[int, transform_lib.Cluster]:
        percentile = percentile * 100
        for k in clusters.keys():
            clusters[k].mean_t = np.mean(data_t[clusters[k].mask,:], axis=0)
            clusters[k].max_t = np.max(data_t[clusters[k].mask,:], axis=0)
            clusters[k].min_t = np.min(data_t[clusters[k].mask,:], axis=0)
            clusters[k].low_perc_t = np.percentile(data_t[clusters[k].mask,:], axis=0, q=percentile)
            clusters[k].high_perc_t = np.percentile(data_t[clusters[k].mask,:], axis=0, q=100-percentile)
        
        return clusters

    @print_time("compute clusters")
    def get_clusters(self, data : npt.NDArray, cluster_engine : ClusterMixin, outliers_mask : npt.NDArray, no_neg_mask : npt.NDArray) -> npt.NDArray:
        labels = np.zeros_like(no_neg_mask, dtype=int)
        labels = np.ones(data.shape[0]) * (-1)
        labels[~outliers_mask & no_neg_mask] = cluster_engine.fit_predict(data[~outliers_mask & no_neg_mask])
        labels[np.logical_not(no_neg_mask)] = np.max(labels) + 1 # negative control range is last cluster
        return labels

    @print_time("predict cluster labels")
    def predict_cluster_labels(self,data : npt.ArrayLike,
                               clusters : Dict[int, transform_lib.Cluster],
                               eps : float,
                               zero_dimensions : npt.ArrayLike) -> Dict[int, transform_lib.Cluster]:
        """Predict labels for eac cluster and store them in the cluter dictionary

        Args:
            data (array_like): All data points
            clusters (Dict[int, Cluters]): Dict containing the clusters
            eps (float): factor for the minimal disance relative to the max cluter means
            zero_dimensions (npt.ArrayLike): Indicator array for zero dimensions

        Returns:
            np.ndarray: Indicators for diseases present in each cluster
        """
        
        clusters_tmp = clusters.copy()
        
        clusters_tmp.pop(-1, None) # get rid of outliers
        dim = data.shape[1]
        
        def in_range(c1 : transform_lib.Cluster, c2 : transform_lib.Cluster, max_scale : npt.NDArray) -> npt.NDArray:
            """Compute whether c2 in in the range of c1
            
            Args:
                c1 (transform_lib.Cluster): Reference cluster
                c2 (transform_lib.Cluster): Cluters to be decieded
                max_dists (npt.NDArray): Maximal distances in each dimension

            Returns:
                npt.NDArray: probability of c2 being in range of c1
            """
            
            dist = c1.mean_t - c2.mean_t
            
            # for each in row l we have the distances scaled by distance l
            scaled_avg_dist = np.einsum('k,l -> lk', dist , np.ones_like(dist))

            smoothing = 6
            interval = 0.2
            center = 1.1
            # the - 0.05 helps in case of strong corrlation
            scaled_avg_dist_ = ((np.abs(scaled_avg_dist + 0.4) - center) * smoothing) / interval
            
            in_range = 1 / (1 + np.exp(scaled_avg_dist_))
            np.fill_diagonal(in_range, 1)

            in_range = np.prod(in_range, axis=1)

            return in_range
        
        def greater_than(c1 : transform_lib.Cluster, c2 : transform_lib.Cluster, eps : float, max_dists : npt.NDArray) -> npt.NDArray:

            # larger in the corresponding dimension
            dist = c1.mean_t - c2.mean_t
            dist_scaled = dist / (max_dists + 1e-15)
            
            interval = 0.5
            dist_scaled_ = ((dist_scaled - eps) / interval * 8)

            gt = 1 / (1 + np.exp(-dist_scaled_))

            return gt
            
        

        D = np.zeros(dim)

        j_i = itertools.combinations(clusters_tmp.keys(),2)
        for (i,j) in j_i:
            dists = clusters_tmp[j].mean_t - clusters_tmp[i].mean_t
            D = np.maximum(D, dists)
            
        # let A_jid denote the directed distance cluster_j - cluster_i between midpoints in dimension d
        # let D_d denote the maximal distance between two clusters
        # cluster_i is considered active in dimension d iff
        # \exists j : A_ijd > eps * D_d and A_ijd <= eps * D_d' \forall d' \neq d)
        # in words this requres c_i to be lareger in dimension c_j by at least some sparating distace
        #       and it requires c_i to be in the range of c_j in all other dimensions
        j_i = itertools.combinations(clusters_tmp.keys(),2)
        for j,i in j_i:
            i_gt_j = greater_than(clusters_tmp[i], clusters_tmp[j], eps, D)
            j_in_range = in_range(clusters_tmp[i], clusters_tmp[j],D)
            prob = j_in_range * i_gt_j
            clusters[i].active_probs = np.maximum(clusters[i].active_probs,prob) * (1 - zero_dimensions)

            j_gt_i = greater_than(clusters_tmp[j], clusters_tmp[i], eps, D)
            i_in_range = in_range(clusters_tmp[j], clusters_tmp[i],D)
            prob = i_in_range * j_gt_i
            clusters[j].active_probs = np.maximum(clusters[j].active_probs,prob) * (1 - zero_dimensions)
            
        return clusters
    
    
    def assign_true_cluster_labels(self, df_gt : pd.DataFrame):
        clusters = self.cluster_dict
        for c in clusters.keys():
            df_lab = df_gt.loc[clusters[c].mask, self.prediction_axis]
            np_lab = np.array(df_lab,dtype=bool)
            n_pt = clusters[c].n
            active = np.sum(np_lab, axis=0) >= n_pt / 2
            clusters[c].active = active
