import numpy as np
import numpy.typing as npt
from typing import Tuple
from sklearn.ensemble import IsolationForest
from scipy import stats as sp
from icecream import ic
    
def get_negative_dimensions(np_points : npt.NDArray,
                            outliers_percentile : float = 0.001,
                            ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Get the dimensions in which the sample `np_point` is not contaminated
    
    This works under the assumption that points are normally distributed
    in dimension in which all are negative and not normally distributed in dimensions,
    where all some positive.
    
    Under this assumption we make a normaltest
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    
    According to experiments performed on each chamber on the given data available for coding,
    we found that dimensions in which we found only a negative cluster the test statistic takes
    in average and a standard deviation (over all samples and dimensions
    in which a dimension only contains a negative cluster) of
    mean: 3247
    std: 2657
    As opposed to samples and dimensions in which there exist positive droples:
    mean: 32887
    std: 7189
    
    By this clear distinction, it is reasonable to distinguish based on this statistic.
    We consider samples (dimension in one sample) to have a probability of 0.99
    that they correspond to the given class (all negative, some positive).
    Then we further assume that in the middle of the two means we have a
    probability of 0.5. Using this, we then fit a sigmoid and return the result
    for a new cluster.

    Args:
        np_points (npt.NDArray): points to be inspected
        acceptable_contamination (float, optional): Outliers which can be contaminated points
            in a negative control for example. Defaults to 0.001.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: Probability for the given dataset to have positive points in this dimension
            test statistic values.
    """
    outlier_detector = IsolationForest(contamination=outliers_percentile,
                                        n_jobs=3,
                                        max_samples=np_points.shape[0],
                                        n_estimators=10)
    outliers_labels = outlier_detector.fit_predict(np_points)
    outliers_mask = outliers_labels < 0
    np_points_no_outlier = np_points[~outliers_mask]
    
    
    # we consider a dimension to be zero if the 1 - maximal_expected_contamination
    # covers more than (1 - maximal_expected_contamination) / 10 of the range (max - min)
    s_max = np.max(np_points_no_outlier, axis=0)
    s_min = np.min(np_points_no_outlier, axis=0)
    r = s_max - s_min

    statistic = sp.normaltest(np_points_no_outlier / r).statistic
    
    # this can be tuned to fit a different certainity curve
    mean_statistic_negative = 3250
    mean_statistic_positive = 32900
    certainity_at_means = 0.99
    
    distance = mean_statistic_positive - mean_statistic_negative
    def inv_sigmoid(y):
        return np.log(y / (1-y))
    sigmoid_distance = inv_sigmoid(certainity_at_means) - inv_sigmoid(1-certainity_at_means)

    center = (mean_statistic_positive + mean_statistic_negative) / 2
    scale = distance / sigmoid_distance
    certainty = 1 / (1 + np.exp(-(statistic - center) / scale))
    
    return certainty, statistic