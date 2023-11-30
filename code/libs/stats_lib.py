import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
from sklearn import metrics


def compute_results(df_probabilites : pd.DataFrame, min_threshhold : float ,max_treshhold : float, df_data : pd.DataFrame) -> pd.DataFrame:
    """Compute statistics on given results. For each column in df_prbabilities the following statistics are computed

    - Concentration: -4440 * log_10(1 - (#positve droplets) / (#total droplets))
    - Number of negative Droplets
    - Number of positive Droplets
    - Separability score  Davies-Bouldin Index [1]
    - Concentration Stock Min, same as Concentration, just with a higher threshhold for points to be considered positive
    - Concentration Stock Max, same as Concentration, just with a lower threshhold for points to be considered positive
    - Concentration relative uncertainty: (ConcentrationMaxStock - ConcentrationMinStock) / Concentration / 2
    
    [1]: D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. PAMI-1, no. 2, pp. 224-227, April 1979, doi: 10.1109/TPAMI.1979.4766909.

    Args:
        df_probabilites (pd.DataFrame): Dataframe with columns corresponding to diseases
        min_threshhold (float): Threshhold for Concentration Stock Max
        max_treshhold (float): Threshhold for Concentration Stock Min
        df_data (pd.DataFrame): Data used for separability score (without labels)

    Returns:
        pd.DataFrame : Dataframe with the above mentioned statistics
    """

    assert (df_probabilites.columns == df_data.columns).all()
    
    df_results = {}
    
    for disease in df_data.columns:

        np_probs = np.array(df_probabilites.loc[:,disease])
        threshhold = (max_treshhold + min_threshhold) / 2
        num_tot, min_pos, _ = compute_pos_neg(np_probabilities=np_probs, theshhold=min_threshhold)
        _, mid_pos, mid_neg = compute_pos_neg(np_probabilities=np_probs, theshhold=threshhold)
        _, max_pos, _ = compute_pos_neg(np_probabilities=np_probs, theshhold=max_treshhold)
        

        concentration_max_stock = compute_concetration(num_tot, min_pos)
        concentration_min_stock = compute_concetration(num_tot, max_pos)
        concentration = compute_concetration(num_tot, mid_pos)
        
        relative_uncertainty = compute_relative_uncertainty(concentration_min_stock, concentration_max_stock)
        
        np_data = np.array(df_data)
        labels = np_probs > threshhold
        ch_score, bd_score = compute_separability_score(np_data, labels)

        df_results[compute_name(disease, 'Concentration')] = [concentration]
        df_results[compute_name(disease, 'NumberOfPositiveDroplets')] = [mid_pos]
        df_results[compute_name(disease, 'NumberOfNegativeDroplets')] = [mid_neg]
        df_results[compute_name(disease, 'SeparabilityScore')] = [bd_score]
        df_results[compute_name(disease, 'ConcentrationStock_Min')] = [concentration_min_stock]
        df_results[compute_name(disease, 'ConcentrationStock_Max')] = [concentration_max_stock]
        df_results[compute_name(disease, 'ConcentrationStock_RelativeUncertainty')] = [f'{relative_uncertainty * 100}%']
        
    df_results = pd.DataFrame.from_dict(df_results)

    return df_results
        

def compute_name(prefix : str, name :str) -> str:
    return f'{prefix}_{name}'
        

def compute_pos_neg(np_probabilities : npt.NDArray, theshhold : float) -> Tuple[int, int, int] :
    """Compute the number of points for ONE class

    Args
        np_probabilities (npt.NDArray): prbabilities for each points to belong to the class
        theshhold (float): above this a point is considered positive

    Returns:
        Tuple[int, int, int]: total number of points, #positives, #negatives
    """
    assert len(np_probabilities.shape) == 1

    num_tot = int(np_probabilities.shape[0])
    num_pos = int(np.sum(np_probabilities >= theshhold))
    num_neg = num_tot - num_pos
    return num_tot, num_pos, num_neg
    
def compute_concetration(num_tot : int, num_pos : int, scale : float = 4440.) -> float:
    """Concentration: -4440 * log_10(1 - (#positve droplets) / (#total droplets))
    
    Note the number 4440 is some density to adjust for previous scores

    Args:
        num_tot (int): # points in total
        num_pos (int): # positive points
        scale (float, optional): Denstiy. Defaults to 4440..

    Returns:
        float: Concentration described above.
    """

    concentration = - scale * np.log10(1 - (num_pos / num_tot)) 
    return concentration
    
def compute_relative_uncertainty(concentration_min_stock : float, concentration_max_stock : float):
    """Compute relative uncertainty:
    
    Concentration relative uncertainty: (ConcentrationMaxStock - ConcentrationMinStock) / Concentration / 2

    Args:
        concentration_min_stock (float):
        concentration_max_stock (float): 

    Returns:
        float: relative uncertainty
    """
    concentration_diff = (concentration_max_stock - concentration_min_stock)
    concentration = (concentration_max_stock + concentration_min_stock) / 2
    uncertainty = concentration_diff / concentration / 2
    return uncertainty


def compute_separability_score(np_data : npt.NDArray, np_labels : npt.NDArray) -> Tuple[float, float]:
    """Separability score between the two classes of positive and negative points
    (In theory works for more classes but the use here is only for the two!)

    - Calinski-Harabasz Index:
        Caliński, Tadeusz & JA,
        Harabasz. (1974). A Dendrite Method for Cluster Analysis. Communications in Statistics - 
        Theory and Methods. 3. 1-27. 10.1080/03610927408827101. 
    
    -  Davies-Bouldin Index:
        D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. PAMI-1, no. 2, pp. 224-227, April 1979, doi: 10.1109/TPAMI.1979.4766909.

    Args:
        np_data (npt.NDArray): data points
        np_labels (npt.NDArray): corresponding 1 and 0 labels

    Returns:
        Tuple[float, float]: The two methods above it the listed order
    """
    
    chs_score = metrics.calinski_harabasz_score(np_data,np_labels)
    db_score = metrics.davies_bouldin_score(np_data, np_labels)
    
    return chs_score, db_score