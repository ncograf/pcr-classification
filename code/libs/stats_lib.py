import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
from sklearn import metrics
from icecream import ic


def compute_results(df_probabilites : pd.DataFrame, threshold : float, df_data : pd.DataFrame) -> pd.DataFrame:
    """Compute statistics on given results. For each column in df_prbabilities the following statistics are computed

    - Concentration: -4440 * log_10(1 - (#positve droplets) / (#total droplets))
    - Number of negative Droplets
    - Number of positive Droplets
    - Separability score  Davies-Bouldin Index [1]
    - Concentration Stock Min, same as Concentration, but we also take uncertainties of points with probabilities > threshold into account
    - Concentration Stock Max, same as Concentration, but we also take uncertainties of points with probabilities < threshold into account
    - Concentration relative uncertainty: (ConcentrationMaxStock - ConcentrationMinStock) / Concentration / 2
    
    [1]: D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. PAMI-1, no. 2, pp. 224-227, April 1979, doi: 10.1109/TPAMI.1979.4766909.

    Args:
        df_probabilites (pd.DataFrame): Dataframe with columns corresponding to diseases
        threshold (float): Threshold for positive and negative labels
        df_data (pd.DataFrame): Data used for separability score (without labels)

    Returns:
        pd.DataFrame : Dataframe with the above mentioned statistics
    """

    assert (df_probabilites.columns == df_data.columns).all()
    
    df_results = {}
    
    for disease in df_data.columns:

        np_probs = np.array(df_probabilites.loc[:,disease])
        
        # ignore outliers
        outlier = np_probs < 0
        np_probs = np_probs[~outlier]
        num_tot, mid_pos, min_pos, max_pos = compute_pos_min_max(np_probabilities=np_probs, threshold=threshold)

        concentration_max_stock = compute_concetration(num_tot, max_pos)
        concentration_min_stock = compute_concetration(num_tot, min_pos)
        concentration = compute_concetration(num_tot, mid_pos)
        
        relative_uncertainty = compute_relative_uncertainty(concentration_min_stock, concentration_max_stock)
        
        np_data = np.array(df_data)[~outlier]
        labels = np_probs > threshold
        ch_score, bd_score = compute_separability_score(np_data, labels)

        df_results[compute_name(disease, 'Concentration')] = [concentration]
        df_results[compute_name(disease, 'NumberOfPositiveDroplets')] = [mid_pos]
        df_results[compute_name(disease, 'NumberOfNegativeDroplets')] = [num_tot - mid_pos]
        df_results[compute_name(disease, 'SeparabilityScore')] = [bd_score]
        df_results[compute_name(disease, 'ConcentrationStock_Min')] = [concentration_min_stock]
        df_results[compute_name(disease, 'ConcentrationStock_Max')] = [concentration_max_stock]
        df_results[compute_name(disease, 'ConcentrationStock_RelativeUncertainty')] = [f'{relative_uncertainty * 100}%']
        
    df_results = pd.DataFrame.from_dict(df_results)

    return df_results

def compute_short_results(df_probabilites : pd.DataFrame, threshold : float, df_data : pd.DataFrame) -> pd.DataFrame:
    """Compute statistics on given results. For each column in df_prbabilities the following statistics are computed

    - Concentration: -4440 * log_10(1 - (#positve droplets) / (#total droplets))
    - Separability score  Davies-Bouldin Index [1]
    - Concentration relative uncertainty: (ConcentrationMaxStock - ConcentrationMinStock) / Concentration / 2
    
    [1]: D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. PAMI-1, no. 2, pp. 224-227, April 1979, doi: 10.1109/TPAMI.1979.4766909.

    Args:
        df_probabilites (pd.DataFrame): Dataframe with columns corresponding to diseases
        threshold (float): Threshold for positive and negative labels
        df_data (pd.DataFrame): Data used for separability score (without labels)

    Returns:
        pd.DataFrame : Dataframe with the above mentioned statistics
    """
    all_results = compute_results(df_probabilites, threshold, df_data)
    concentration = []
    bd_score = []
    positives = []
    negatives = []
    rel_uncetainty = [] 
    for col in df_probabilites.columns:
        positives.append(all_results.loc[0, compute_name(col, 'NumberOfPositiveDroplets')])
        negatives.append(all_results.loc[0, compute_name(col, 'NumberOfNegativeDroplets')])
        concentration.append(all_results.loc[0, compute_name(col, 'Concentration')])
        bd_score.append(all_results.loc[0, compute_name(col, 'SeparabilityScore')])
        rel_uncetainty.append(all_results.loc[0, compute_name(col, 'ConcentrationStock_RelativeUncertainty')])

    df_all = {}
    df_all["Disease"] = df_probabilites.columns
    df_all["#Positives"] = positives
    df_all["#Negatives"] = negatives
    df_all["Concentration"] = concentration
    df_all["Separability"] = bd_score
    df_all["Uncertainty"] = rel_uncetainty
    df_all = pd.DataFrame.from_dict(df_all)

    return df_all
        

def compute_name(prefix : str, name :str) -> str:
    return f'{prefix}_{name}'
        

def compute_pos_min_max(np_probabilities : npt.NDArray, threshold : float) -> Tuple[int, int, int] :
    """Compute the number of points for ONE class (also compute min and max stock)

    Args
        np_probabilities (npt.NDArray): prbabilities for each points to belong to the class
        theshhold (float): above this a point is considered positive

    Returns:
        Tuple[int, int, int]: total number of points, #positives, #negatives
    """
    assert len(np_probabilities.shape) == 1

    mask = np_probabilities >= threshold

    num_tot = int(np_probabilities.shape[0])
    num_pos = int(np.sum(mask))

    # also add the uncertainty in the netatives which might be positive
    num_max_pos = int(np.sum(mask) + np.sum(np_probabilities[~mask]))

    # also consider only uncertainty of positives
    num_min_pos = int(np.sum(np_probabilities[mask]))

    return num_tot, num_pos, num_min_pos, num_max_pos
    
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
    uncertainty = concentration_diff / (concentration * 2 + 1e-15)
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
    
    db_score = np.nan
    chs_score = np.nan
    if len(np.unique(np_labels)) > 1:
        chs_score = metrics.calinski_harabasz_score(np_data,np_labels)
        db_score = metrics.davies_bouldin_score(np_data, np_labels)
    
    return chs_score, db_score