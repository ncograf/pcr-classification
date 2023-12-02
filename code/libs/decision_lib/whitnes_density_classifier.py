import numpy as np
import pandas as pd
import scipy.stats as sp
import numpy.typing as npt
import transform_lib
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Callable, ParamSpec, TypeVar
from icecream import ic
import warnings
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
                 negative_control : npt.ArrayLike,
                 cluster_algorithm : ClusterMixin,
                 whitening_transformer : TransformerMixin,
                 eps : float = 1.7,
                 contamination : float = 0.01,
                 negative_range: float = 0.9,
                 outliers : bool = True,
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
        self.eps = eps
        self.contamination = contamination
        self.get_outlier = outliers
        self.verbose = verbose
        self.thresh = 0.5

        
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
        
        # remove points in negative control range
        self.X = X
        X_no_neg = self.negative_remover.transform(X)
        self.No_neg_mask = self.negative_remover.mask
            
        # compute zero dimensions
        self.neg_dimensions, outliers = self.get_negative_dimensions(self.X, acceptable_contamination=0.001, maximal_expected_contamination=0.4)
        
        # remove outliers and create clusters
        cluster_labels_no_neg = self.get_clusters_and_outlier(X_no_neg,
                                                              self.cluster_algorithm,
                                                              contamination=self.contamination,
                                                              get_outliers=self.get_outlier)

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
        
        # make cluster predictions
        self.cluster_dict = self.predict_cluster_labels(self.X_transformed,
                                                        clusters=self.cluster_dict,
                                                        eps=self.eps,
                                                        zero_dimensions=self.neg_dimensions)

        # generate label predicitions
        self.probabilities, self.predictions = self.predict_labels(clusters=self.cluster_dict, data=self.X, thresh=self.thresh)
        self.predictions[self.cluster_labels < 0,:] = -1
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = self.predictions, columns=self.prediction_axis)
        self.probabilities_df = pd.DataFrame(data = self.probabilities, columns=self.prediction_axis)
    
        return self.predictions_df
    
    
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
        n_clusters = len(clusters_tmp)
        dim = data.shape[1]

        # A_{i,j,d} =  cluster_i - cluster_j in dimension d (where we compare midpoints)
        A = np.ones((n_clusters,n_clusters,dim), dtype=float)
        
        def in_range(c1 : transform_lib.Cluster, c2 : transform_lib.Cluster, max_scale : npt.NDArray, d : int) -> npt.NDArray:
            """Compute whether c2 in in the range of c1
            
            Args:
                c1 (transform_lib.Cluster): Reference cluster
                c2 (transform_lib.Cluster): Cluters to be decieded
                max_dists (npt.NDArray): Maximal distances in each dimension
                d (int): relevant dimension

            Returns:
                float: probability of c2 being in range of c1
            """
            
            dist = c1.mean_t - c2.mean_t
            smoothing = 0.6
            scaled_avg_dist = dist / (dist[d] + 1e-15) * smoothing

            # we want the sigmoid function to operate on [-8,8] scaled_avg_dist is in the range
            # interbal around mean so we get
            interval = 0.2
            mean = smoothing * 1.3
            scaled_avg_dist_ = (np.abs(scaled_avg_dist) - mean) / interval * 8
            
            in_range = 1 / (1 + np.exp(scaled_avg_dist_))
            
            d_mask = np.ones(dim, dtype=bool)
            d_mask[d] = 0
            
            in_range = np.prod(in_range[d_mask])

            return in_range
        
        def greater_than(c1 : transform_lib.Cluster, c2 : transform_lib.Cluster, eps : float, max_dists : npt.NDArray, d : int) -> bool:

            # larger in the corresponding dimension
            dist = c1.mean_t - c2.mean_t
            dist_scaled = dist / (max_dists + 1e-15)
            
            interval = 1
            dist_scaled_ = ((dist_scaled - eps) / interval * 6)[d]

            gt = 1 / (1 + np.exp(-dist_scaled_))

            return gt
            
        
        for j in clusters_tmp.keys():
            for i in clusters_tmp.keys():
                # mean_t is in the transformed data
                dists = clusters_tmp[j].mean_t - clusters_tmp[i].mean_t
                A[j,i] = dists
        
        D = np.max(A, axis=(0,1))
        D = np.max(A, axis=(0,1))
        
        # let A_jid denote the directed distance cluster_j - cluster_i between midpoints in dimension d
        # let D_d denote the maximal distance between two clusters
        # cluster_i is considered active in dimension d iff
        # \exists j : A_ijd > eps * D_d and A_ijd <= eps * D_d' \forall d' \neq d)
        # in words this requres c_i to be lareger in dimension c_j by at least some sparating distace
        #       and it requires c_i to be in the range of c_j in all other dimensions
        for j in clusters_tmp.keys():
            for i in clusters_tmp.keys():
                for d in range(dim):
                    if not zero_dimensions[d]:
                        j_in_range = in_range(clusters_tmp[i], clusters_tmp[j],D, d)
                        i_gt_j = greater_than(clusters_tmp[i], clusters_tmp[j], eps, D, d)
                        prob = j_in_range * i_gt_j
                        if prob > clusters[i].active_probs[d]:
                            clusters[i].active_probs[d] = prob
        
        return clusters
    
    
    def assign_true_cluster_labels(self, df_gt : pd.DataFrame):
        clusters = self.cluster_dict
        for c in clusters.keys():
            df_lab = df_gt.loc[clusters[c].mask, self.prediction_axis]
            np_lab = np.array(df_lab,dtype=bool)
            n_pt = clusters[c].n
            active = np.sum(np_lab, axis=0) >= n_pt / 2
            clusters[c].active = active

    @print_time("negative dimensions")
    def get_negative_dimensions(self,np_points : npt.NDArray,
                                acceptable_contamination : float = 0.001,
                                maximal_expected_contamination : float = 0.4) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the dimensions in which the sample `np_point` is not contaminated
        and as a bonuns get outliers
        
        How can we do this: The fundamental Idea is that most of the points are negative
        in all dimensions (according the data we that was avilable while constructing
        this algorithm the procentage of such points was between 85 and 100).
        Moreover, the second obervation on which this algorithm is based, is that these
        point (which are negative in all dimensions) do not have a large spread 
        (in the data the spread of the negative data was around 10 - 20 percent of
        the whole range of point values in samples with positively lavelled points)
        
        First the algorithm removes outliers and in fact it assumes to have
        the precentile `acceptable contamination` of points to be outliers.
        After this we call the sample without the outliers S.

        We use the parameter
        -> `maximal_expected_contamination` = mec
        int the following
        
        For a sample S, lets only condier axis d and let r := max(S) - min(S) (in axis d).
        Further let q \in \R be such that |{p \in S : p_d - min(S_d) < q}| = (1 - mec)|S|.
        (The intuition here is that at least all points below q will be negatives).
        Then we consider the axis d of S to be completely negative if
        (q / r) >= (1 - mec) / 2.
        
        So we need the following assumptions on a sample S for this to work:
            - In each dimension, less than `maximal_expected_contamination` * |S| points are
                contaminated with the disease corresponding to dimension d.
            - The distribution of the "negative" points in dimension d of sample |S|
                is evenly distributed, more specifically: let N(S) be all "negative" points
                of S in dimension d. Then let q (as above) be the span such that (1-mec)
                points are contained in a range q. These points must then span more than
                (1-mec) / 2 times the range spanned by N(S) in dimension d. 
                Note that for uniformly distributed data we would expect
                them to span (1-mec) the range and for normally distributed
                data more that 0.5 as long as mec < 0.5. Here we expect 
                the algorithm will work better the higher the procentile of 
                netative points. Althought there would needs to be a 300% more positive
                points such that this would have a serious negative effect.
            - Lastly we assume that positive samples in dimension d have a higher value
                than min(S_d) + 2 * p / (1 - mec). This is again a reasonable assumption
                based on the data we have, as we have this value higher than 
                min(S_d) + 6 * p / (1 - mec). 
        

        Args:
            np_points (npt.NDArray): points to be inspected
            acceptable_contamination (float, optional): Outliers which can be contaminated points
                in a negative control for example. Defaults to 0.001.
            maximal_expected_contamination (float, optional): Over all samples in a series
                there should not be more than a quantile of this amount conatimated. Defaults to 0.4.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: Indicator array for each dimension to be only negative, 
                outplier labels (-1 are outliers).
        """
        outlier_detector = IsolationForest(contamination=acceptable_contamination,
                                            n_jobs=3,
                                            max_samples=np_points.shape[0],
                                            n_estimators=10)
        outliers_labels = outlier_detector.fit_predict(np_points)
        np_points_no_outlier = np_points[outliers_labels >= 0]
        
        # we consider a dimension to be zero if the 1 - maximal_expected_contamination
        # covers more than (1 - maximal_expected_contamination) / 10 of the range (max - min)
        s_max = np.max(np_points_no_outlier, axis=0)
        s_min = np.min(np_points_no_outlier, axis=0)
        r = s_max - s_min

        mec = maximal_expected_contamination
        q = np.percentile(np_points_no_outlier, 100 - 100 * mec, axis=0)
        q = q - s_min
        
        # consider every dimensin to be zero as above
        zero_dimensions = (q / r) * 2 >= (1 - mec)
            
        return zero_dimensions, outliers_labels