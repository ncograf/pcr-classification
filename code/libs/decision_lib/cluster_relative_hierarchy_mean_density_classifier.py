import numpy as np
import pandas as pd
import scipy.stats as sp
import numpy.typing as npt
import transform_lib
import itertools
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple
from icecream import ic
from enum import Enum

class Comparator(Enum):
        MUCHSMALLER = -2
        SMALLER = -1
        EQUAL = 0
        LARGER = 1
        MUCHLARGER = 2

class ClusterRelativeHierarchyMeanDensityClassifier(BaseEstimator):

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
        
        # make cluster predictions ==> use non-tranformed data
        self.cluster_dict = self.predict_cluster_labels(self.X, clusters=self.cluster_dict, eps=self.eps)

        # generate label predicitions
        self.predictions, self.probabilies = self.predict_labels(clusters=self.cluster_dict, data=self.X)
        self.predictions[self.cluster_labels < 0,:] = -1
        self.probabilies[self.cluster_labels < 0,:] = -1
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = self.predictions, columns=self.prediction_axis)
        self.probabilies_df = pd.DataFrame(data = self.probabilies, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def predict_cluster_labels(self,data : npt.ArrayLike, clusters : Dict[int, transform_lib.Cluster], eps : float) -> Dict[int, transform_lib.Cluster]:
        """Predict labels for eac cluster and store them in the cluter dictionary
        
        Sideffects:
            - updates cluster dictionarry internally such that output is redundant with input after computation

        Args:
            data (array_like): All data points
            clusters (Dict[int, Clusters]): Dict containing the clusters
            eps (float): factor for the minimal disance relative to the max cluter means
        
        Returns:
            Dict[int, transform_lib.Cluster] : Input which was updated along the computations
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

        _ = self.compute_comparators(clusters_tmp, dimensions, eps)
        _ = self.compute_hierarchy(clusters_tmp, dim=dim)
        _ = self.select_max_dimensions(clusters_tmp, dimensions)

        for i in clusters_tmp.keys():
            clusters[i] = clusters_tmp[i]
        
        return clusters
    
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

        probabilites = np.zeros_like(data, dtype=np.float32)
        for c in clusters.keys():
            probabilites[clusters[c].mask] = clusters[c].active_probs

        return predictions, probabilites

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

    def get_clusters_and_outlier(self, data : npt.ArrayLike, cluster_engine : ClusterMixin, get_outliers=True, contamination = 0.001) -> npt.NDArray:

        if get_outliers:
            outlier_detector = IsolationForest(contamination=contamination,
                                               n_jobs=4,
                                               max_samples=data.shape[0],
                                               n_estimators=10)
            labels = outlier_detector.fit_predict(data) # outliers will get label -1
            print(len(labels[labels == -1]), " outliers detected")
            labels[labels >= 0] = cluster_engine.fit_predict(data[labels >= 0])
        else:
            labels = cluster_engine.fit_predict(data)
        
        return labels
    
    def select_max_dimensions(self, clusters : Dict[int, transform_lib.Cluster], dimensions_scale : np.ndarray) -> Dict[int, transform_lib.Cluster]:
        """
        Given that we know the number of dimensions the clusters are active to classify them we only select the number of dimensions
        they are active in
        
        Sideffects:
            - Updates clusters internally, adds / changes a variable `hierarchy_porbs`

        Args:
            clusters (Dict[int, Cluters]): Dict containing the clusters
            dimension_scale (float): min-max for each dimension across all clusters
        """
        for c in clusters.keys():

            cluster = clusters[c]
            
            # compute the relatively largest dimensions
            order = np.flip(np.argsort(cluster.mean / dimensions_scale))
            cluster.labels[order] = cluster.hierarchy_probs >= 0.5
            cluster.active_probs[order] = cluster.hierarchy_probs

        return clusters

    def compute_hierarchy(self, clusters : Dict[int, transform_lib.Cluster], dim : int) -> Dict[int, transform_lib.Cluster]:
        """Compute the hierarchy based on the binary comparisons

        Sideffects:
            - Updates clusters internally, adds / changes a variable `hierarchy_porbs`

        Args:
            clusters (Dict[int, transform_lib.Cluster]): clusters with internally stored comparisons
            dim (int) : number of dimensions

        Returns:
            Dict[int, transform_lib.Cluster]: Cluster dictionary with stored hieraries (same as input after computation)
        """
    
        # consider all pairs of clusters
        pairs = list(itertools.combinations(clusters.keys(), 2))
    
        # we define rank to refer to the number of active features
        for rank in range(dim):
            for (key_A, key_B) in pairs:
                for d in range(dim):
                
                    d_mask = np.zeros(dim).astype(bool)
                    d_mask[d] = True

                    cluster_A = clusters[key_A]
                    cluster_B = clusters[key_B]

                    A_geq_B = cluster_A.comparotor_probs[key_B]
                    B_geq_A = cluster_B.comparotor_probs[key_A]
                    
                    if rank == 0:
                        # probability of X larger than Z in dimension d and not smaller in all other dimensions
                        potential_prob_A = A_geq_B[d_mask] * np.prod((1 - B_geq_A[~d_mask]))
                        potential_prob_B = B_geq_A[d_mask] * np.prod((1 - A_geq_B[~d_mask]))
                    else:
                        # probability of X larger than Z in dimension d and not smaller in all other dimensions
                        potential_prob_A = A_geq_B[d_mask] * np.prod((1 - B_geq_A[~d_mask]))
                        potential_prob_B = B_geq_A[d_mask] * np.prod((1 - A_geq_B[~d_mask]))

                        # times the probability that X is active in rank - 1
                        potential_prob_A = potential_prob_A * (cluster_A.hierarchy_probs[rank - 1])
                        potential_prob_B = potential_prob_B * (cluster_B.hierarchy_probs[rank - 1])
                        
                        # times the probability that Z is acitve in rank - 1
                        potential_prob_A = potential_prob_A * (cluster_B.hierarchy_probs[rank - 1])
                        potential_prob_B = potential_prob_B * (cluster_A.hierarchy_probs[rank - 1])

                    cluster_A.hierarchy_probs[rank] = max(cluster_A.hierarchy_probs[rank], potential_prob_A)
                    cluster_B.hierarchy_probs[rank] = max(cluster_B.hierarchy_probs[rank], potential_prob_B)
            
        for c in clusters.keys():
            np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
            #ic(c, clusters[c].hierarchy_probs)

        return clusters
    
    def compute_comparators(self, clusters : Dict[int, transform_lib.Cluster], dimension_scale : npt.NDArray, eps : float) -> Dict[int, transform_lib.Cluster]:
        """ Compare all clusters in all dimensions and return a matrix
        
        Sideeffects:
            - changes the clusters internally and outputs them

        Args:
            clusters (Dict[int, Cluters]): Dict containing the clusters
            dimension_scale : (max - min) in each dimension
            eps (float): factor for the minimal disance relative to the max cluter means

        Returns:
            all clusters compared with all clusters in all dimensions as a 3d np.ndarray
        """
        for i in clusters.keys():
            for j in clusters.keys():

                self.compare_two_cluster(clusters=clusters, cluster_A_key=i,cluster_B_key=j, eps=eps, dimension_scale=dimension_scale)
    
        return clusters

    def compare_two_cluster(self,
                            clusters : Dict[int,transform_lib.Cluster],
                            cluster_A_key : int,
                            cluster_B_key : int,
                            eps : float,
                            dimension_scale : List[float]) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compare two clusters and for each of the two store the probability of it beeing larger than the other

        The comparations are stored in the clusters itselfs

        Sideeffects:
            - changes the clusters internally and outputs them

        Args:
            clusters (Dict[int,transform_lib.Cluster]): clusters
            cluster_A_key (int): Cluster label of cluster A
            cluster_B_key (int): Cluster label of cluster B
            eps (float) : Relative (to max - min in each dimension) scale to be considered as absolutely greater
            dimension_scale (List[float]): scale (max - min in each dimension)
        """
        
        cluster_A = clusters[cluster_A_key] 
        cluster_B = clusters[cluster_B_key] 
        
        n_dim = len(dimension_scale)
        assert cluster_B.mean.shape == cluster_A.mean.shape, "Cluster life in different spaces" 
        assert cluster_B.mean.shape[0] == n_dim, "Cluster dimensions do not agree with available dimension scales"
        difference = cluster_A.mean - cluster_B.mean
        base = 2.7
        strech = 5.4
        # sigmoid at 0 is 0.5 and sigmoid(base) ~ 0 and sigmoid(-base) ~ 1
        difference_scaled = difference / np.array(dimension_scale) * strech / eps
        assert difference_scaled.shape[0] == n_dim, "Cluster difference computation did not work somehow"
        A_is_greater_probs = 1 / (1 + np.exp(-difference_scaled + base))
        B_is_greater_probs = 1 / (1 + np.exp(difference_scaled + base))
        cluster_A.comparotor_probs[cluster_B_key] = A_is_greater_probs
        cluster_B.comparotor_probs[cluster_A_key] = B_is_greater_probs
        
        return A_is_greater_probs, B_is_greater_probs