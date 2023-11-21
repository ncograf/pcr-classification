import numpy as np
import pandas as pd
import scipy.stats as sp
import numpy.typing as npt
import transform_lib
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from typing import Dict, List, Tuple, Literal
from icecream import ic


class ClusterHierarchyDensityClassifier(BaseEstimator):

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

        # compute extremas
        self.max = np.max(self.X, axis=0)
        self.min = np.min(self.X, axis=0)

        self.max_t = np.max(self.X_transformed, axis=0)
        self.min_t = np.min(self.X_transformed, axis=0)
        
        # add transformed properties to clusters
        self.cluster_dict = self.add_transformed(self.X_transformed, self.cluster_dict)
        
        # make cluster predictions
        self.cluster_dict = self.predict_cluster_labels(self.X_transformed, clusters=self.cluster_dict, eps=self.eps)

        # generate label predicitions
        self.predictions = self.predict_labels(clusters=self.cluster_dict, data=self.X)
        self.predictions[self.cluster_labels < 0,:] = -1

        # add covariances and only keep clusters which have a invertable covariance
        self.cluster_dict = self.get_cluster_covs(self.X, self.cluster_dict)
        
        # compute point probabilities
        self.kernel_density : List[Tuple[KernelDensity]] = self.compute_density(self.cluster_dict, self.X)
        self.probabilities = self.compute_probs(self.kernel_density, self.X)
        self.probabilities[self.cluster_labels < 0,:] = -1
        
        # add labels to predictions here we use the domain knowledge to label predictions
        self.predictions_df = pd.DataFrame(data = self.predictions, columns=self.prediction_axis)
        self.probabilities_df = pd.DataFrame(data = self.probabilities, columns=self.prediction_axis)
    
        return self.predictions_df
    
    def compute_density(self,
                        clusters : Dict[int, transform_lib.Cluster],
                        data : npt.ArrayLike,
                        kernel : Literal['gaussian', 'epanechnikov'] = 'gaussian',
                        relative_bandwidth : float = 0.1,
                        ) -> List[Tuple[KernelDensity]]:
        """Compute distributions for postives and negatives

        Args:
            clusters (Dict[int, transform_lib.Cluster]): Clusters, containing labels
            data (npt.ArrayLike): Data points

        Returns:
            List[Tuple[KernelDensity]]: 
        """
        densites = []
        dim = data.shape[1]
        extremas = self.max - self.min
        
        for d in range(dim):
            pos_data = []
            neg_data = []
            for k in clusters.keys():
                if clusters[k].labels[d]:
                    pos_data.append(data[clusters[k].mask,:])
                else:
                    neg_data.append(data[clusters[k].mask,:])
            pos_data = np.concatenate(pos_data, axis=0)
            neg_data = np.concatenate(neg_data, axis=0)
            pos_density = KernelDensity(kernel='gaussian',
                                        bandwidth=extremas[d] * relative_bandwidth,
                                        leaf_size=pos_data.shape[0] // 5,
                                        rtol=180000,
                                        ).fit(pos_data)
            neg_density = KernelDensity(kernel='gaussian',
                                        bandwidth=extremas[d] * relative_bandwidth,
                                        leaf_size=neg_data.shape[0] // 5,
                                        atol=180000,
                                        ).fit(neg_data)
            densites.append((pos_density, neg_density))

        return densites


    
    def compute_probs(self,
                      densities : List[Tuple[KernelDensity, KernelDensity]],
                      data : npt.ArrayLike) -> npt.NDArray:
        """Get from cluster prediction to point probabilites using the cluster assignments

        Args:
            densities (List[KernelDensity, KernelDensity]]): density estimations
            data (npt.ArrayLike): Data points used for data shape

        Returns:
            NDArray: Predictions on points
        """
        data_prob = np.zeros_like(data, dtype=np.float32)
        dim = data.shape[1]
        
        for d in range(dim):
            ic(d)
            ic(data.shape)
            pos_prob = np.exp(densities[d][0].score_samples(data))
            neg_prob = np.exp(densities[d][1].score_samples(data))
            data_prob[:, d] = np.einsum('k,k->k', pos_prob, 1 / (pos_prob + neg_prob))
        
        return data_prob

    
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
            clusters[k].mean_t = np.mean(data_t[clusters[k].mask,:], axis=0)
            clusters[k].max_t = np.max(data_t[clusters[k].mask,:], axis=0)
            clusters[k].min_t = np.min(data_t[clusters[k].mask,:], axis=0)
        
        return clusters
    
    def get_cluster_covs(self, data : npt.ArrayLike, clusters : Dict[int, transform_lib.Cluster]) -> Dict[int, transform_lib.Cluster]:
        new_clusters = {}
        for k in clusters.keys():
            loc_dat = (data[clusters[k].mask,:] - clusters[k].mean).T
            cov = np.cov(loc_dat)
            try:
                normal = sp.multivariate_normal(clusters[k].mean, cov)
                new_clusters[k] = clusters[k]
                new_clusters[k].cov = cov
                new_clusters[k].dist = normal
            except:
                pass
        return new_clusters
            

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

    def predict_cluster_labels(self,data : npt.ArrayLike, clusters : Dict[int, transform_lib.Cluster], eps : float) -> Dict[int, transform_lib.Cluster]:
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
        dim = data.shape[1]

        # A_{i,j,d} =  cluster_i - cluster_j in dimension d (where we compare midpoints)
        A = np.ones((n_clusters,n_clusters,dim), dtype=float)

        for j in clusters_tmp.keys():
            for i in clusters_tmp.keys():
                # mean_t is in the transformed data
                dists = clusters_tmp[j].mean_t - clusters_tmp[i].mean_t
                A[j,i] = dists
        
        D = np.max(A, axis=(0,1)) - np.min(A, axis=(0,1))
        
        # let A_jid denote the directed distance cluster_j - cluster_i between midpoints in dimension d
        # let D_d denote the maximal distance between two clusters
        # cluster_i is considered active in dimension d iff
        # \exists j : A_ijd > eps * D_d and A_ijd < -eps * D_d' \forall d' \neq d)
        # in words this requres c_i to be lareger in dimension c_j by at least some sparating distace
        #       and it requires c_i to be within the range of c_j for the other dimensios
        for d in range(dim):
            for j in clusters_tmp.keys():
                for i in clusters_tmp.keys():
                    if A[i,j,d] > eps * D[d]: # cluster_j <_d cluster_i
                        temp = 1
                        for d_ in range(dim):
                            if d_ == d: 
                                continue
                            if A[i,j,d_] < -1.2 * eps * D[d_]: # cluster_j >=_d' cluster_i
                                temp = 0
                        if temp == 1:
                            clusters[i].labels[d] = 1
        return clusters