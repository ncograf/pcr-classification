import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.cluster as cl
import matplotlib.pyplot as plt
from typing import Dict, List, overload
import itertools
import re
import pathlib
import os

LABELS_LIST = ["IAV-M_POS",
             "IAV-M_NEG",
             "IBV-M_POS",
             "IBV-M_NEG",
             "MHV_POS",
             "MHV_NEG",
             "RSV-N_POS",
             "RSV-N_NEG",
             "SARS-N1_POS",
             "SARS-N1_NEG",
             "SARS-N2_POS",
             "SARS-N2_NEG"]


def load_dataset(labels : List[str] = None, datasets : List[str] = None, datafolder : str = "../Data") -> pd.DataFrame:
    """Loads data into a pandas DataFrame

    Args:
        labels (List[str], optional): Labels to be included in the output available labels are accessable in data_lib.LABELS_LIST. Defaults to None.
        datasets (List[str], optional): Dataset shortcuts of the desired datasets which will be merged into the output (List of available set can be generated with data_lib.explore_datasets(root, verbose=True)). Defaults to None.
        datafolder (str, optional): Root folder where datasets will be found. Defaults to "../Data".

    Returns:
        pd.DataFrame: Contains dataset with six columns containing features and one column for each added label. The datasets are concatenated along rows.
    """
    # get available sets
    available_datasets = explore_datasets(datafolder) 
    
    # build list for load_custom_dataset
    dataset_list = []
    
    # check if a selection for datasets is given
    if datasets == None:
        for short in available_datasets:
            dataset_list.append(available_datasets[short])
    else:
        
        # only add datsets that were explored before
        for short in datasets:
            if short in available_datasets:
                dataset_list.append(available_datasets[short])
            else:
                print(f"Dataset with the short form {short} not available in {datafolder}")
    
    # build list for load_custom_dataset
    label_list = []
    
    # check if label selection is present
    if labels == None:
        label_list = ["IAV-M_POS",
                    "IBV-M_POS",
                    "MHV_POS",
                    "RSV-N_POS",
                    "SARS-N1_POS",
                    "SARS-N2_POS"]
    else:
        
        # only add labels available
        for label in labels:
            if label in LABELS_LIST:
                label_list.append(label)
            else:
                print(f"Label {label} is not available")
            
    # compute and return dataset
    return load_custom_dataset(dataset_list, label_list)
    
def explore_datasets(datafolder : str = "../Data", verbose=False) -> Dict[str, List[str]]:
    """Searches for files with "labelled" datacontent

    Args:
        datafolder (str, optional): root, where the search should start. Defaults to "../Data".
        verbose (bool, optional): If True, summary of found folder will be printed. Defaults to False.

    Returns:
        Dict[str, List[str]]: Dictinary containing found data files, the files are gouped in lists and stored
        under the short name which is computed from the original file name
    """
    
    # generate pattern 
    # must have
    #   .csv
    #   -labelled
    #   _RawData.csv
    dataset_pattern = "6P-(.*)-labelled_.*_(\w\d).*_RawData.csv"
    
    # create final dictionary
    data_dict : Dict[str, List[str]] = {}
    
    # create groups
    data_groups : Dict[str, Dict[str]] = {}

    # explore data folder 
    for (root, _, files) in os.walk(datafolder, topdown=True):
        # search for csv files with the pattern above
        for file in files:

            pattern_match = re.search(dataset_pattern, file)
            if pattern_match:
                # file is a match
                file_path = str(pathlib.Path(root,file))
                
                # generate key and group
                pattern_groups = pattern_match.groups()
                word_start = re.findall(r'\b[\w|\d]{1,2}', pattern_groups[0])
                file_grp = '-'.join(word_start)
                file_short = '-'.join(word_start + [pattern_groups[1]])
                
                # append or create new list
                if file_short in data_dict:
                    data_dict[file_short].append(file_path)
                else:
                    data_dict[file_short] = [file_path]
                
                if file_grp in data_groups:
                    data_groups[file_grp][file_short] = len(data_dict[file_short])
                else: 
                    data_groups[file_grp] = {}
                    data_groups[file_grp][file_short] = len(data_dict[file_short])
                    data_groups[file_grp]["path"] = root
                    
    if verbose:
        print("----------------------------------------------------------------------------------------------")
        print(f"-- The following {len(data_groups)} groups were found")
        print(f"-- They contain {len(data_dict)} datasets")
        print(f"-- The first printed entity is the key to the returned dictionary" )
        for key in data_groups:
            print("-----------------------------------")
            print(f"Group: {data_groups[key]['path']}")
            
            # list for printing
            group_elems = []
            for set_key in data_groups[key]:
                if set_key != 'path':
                    #print(f"{set_key}, files: {data_groups[key][set_key]}, sample path: {data_dict[set_key][0]}")
                    group_elems.append(f"{set_key}, files: {data_groups[key][set_key]}")
            
            # print
            for a,b,c in zip(group_elems[::3], group_elems[1::3], group_elems[2::3]):
                print(f"{a:<40}{b:<40}{c}")
        print("-----------------------------------")

    # Then group files with the same `.*_S-.*_.._` pattern
    # extract their paths and store it in a map with
    return data_dict

# get data to work with
def load_custom_dataset(files: List[List[str]], labels: List[str] ) -> pd.DataFrame:
    """Generate Dataframe with the six features and for each indicated label one more column.

    Args:
        files (List[List[str]]): List of Lists of filenames with data location.
            Each list in the list is treated as one set of data (Must contain RawData and LabeledFiles)
            The first filename in each list which matches the pattern `.*_.._RawData` is considered the raw data
            The rest of the files will be compared to the given labels and selected accordingly on demand
        labels (List[str]): List of labels e.g.
             ["IAV-M_POS",
             "IBV-M_POS",
             "MHV_POS",
             "RSV-N_POS",
             "SARS-N1_POS",
             "SARS-N2_POS"]

    Returns:
        pd.DataFrame: Dataframe with one row for each point found in one of the files, 
            six columns for the features and one column for each label
    """
    if len(files) <= 0:
        raise Exception("Error: No files provided.")

    df_major = None
    for file_list in files:
        
        # find RawData
        raw_data_file_pattern = ".*_.._RawData.csv"
        raw_data_files = list(filter(re.compile(raw_data_file_pattern).findall, file_list))
        if len(raw_data_files) <= 0:
            raise Exception(f"Error: Not all indicated data lists contain `{raw_data_file_pattern}`.")
        
        # pick first file matching the pattern
        raw_data_file = raw_data_files[0]
        if not os.path.isfile(raw_data_file):
            raise FileExistsError(f'Error: invalid file name {raw_data_file}.')
        raw_data = pd.read_csv(raw_data_file)
        
        # ignore whitspaces and cases
        raw_data.rename(columns=lambda x : x.strip(), inplace=True)
    
        # drop coordinate and index columns (if they exists by errors=ignore)
        raw_data.drop(["x-coordinate_in_pixel","y-coordinate_in_pixel","index"],axis=1, errors='ignore', inplace=True)

        
        # get label files
        for label in labels:
            
            # find label file
            label_file_pattern = f"{label}.*.csv"
            label_files = list(filter(re.compile(label_file_pattern).findall, file_list))
            if len(label_files) <= 0:
                raise Exception(f"Error: Not all indicated data lists contain data for label: `{label_file_pattern}`.")

            # pick the first file containing the label
            label_file = label_files[0]
            if not os.path.isfile(label_file):
                raise FileExistsError(f"Error: invalid file name {label_file}.")
            label_data = pd.read_csv(label_file)
            
            # ignore whitspaces and cases
            label_data.rename(columns=lambda x : x.strip(), inplace=True)
            label_data.rename(columns=lambda x : x.lower(), inplace=True)
            
            # get indices of labelled points
            label_indices = label_data['index']
            
            # append row in raw_data with the corresponding labels
            raw_data[label] = 0
            raw_data[label].iloc[label_indices] = 1
        
        # append to df_major if it exists
        if df_major is None:
            df_major = raw_data
        else:
            df_major = pd.concat((df_major, raw_data), axis=0)
    
    return df_major

def pairwise_plots(label : str, data_sets : List[str], classifier : skl.base.ClusterMixin, data_folder : str = "../Data", verbose : bool = False):
    """Plots the clustering pojected onto two dimension

    Args:
        label (str): Selected Label out of LABELS_LIST
        data_sets (List[str]): List of dataset(s) to include in the plot
        classifier (skl.base.ClusterMixin): Classifier to be used for unsupervised clustering
        data_folder (str, optional): Data folder to explore datasets. Defaults to "../Data".
        verbose (bool, optional): If true prints information about found clusters and hitting rates for labels. Defaults to False.
    """
    
    df = load_dataset([label], data_sets, data_folder)
    
    # fist six columns are features
    df_features = df.iloc[:, :6]
    np_features = df_features.to_numpy(copy=True)
    feature_names = df_features.columns
    
    # the sixth column is interpreded as label
    df_labels = df.iloc[:, 6]
    np_labels = df_labels.to_numpy(copy=True)
    label_name = df_labels.name
    
    # predict
    preds = classifier.fit_predict(np_features)
    
    # count clusters and label outliers
    clusters_all = np.unique(preds)
    clusters = np.delete(clusters_all, clusters_all == -1)
    outliers = (preds == -1)

    # define markers for plot
    marker_map = [".", "p", "+", "v", "^", ">", "<", "1", "2", "3", "4", "8", "s", "p", "P", "*"]
    cluster_marker_index = clusters_all % len(marker_map)
    cluster_marker = [marker_map[i] for i in cluster_marker_index]
    
    # create label_match array -1 outliers, 0 true pos, 1 false pos, 2 true neg, 3 false neg
    label_match_legend = {"outlier":-1, "true pos":0, "false pos":1, "true neg":2, "false neg":3}
    label_color_map = {-1:"black", 0:"green", 1:"red", 2:"lime", 3:"magenta"}
    label_match = np.ones_like(preds) * -1 # default outlier
    
    # for each cluster associate true of false
    for cluster in clusters:
        
        # select elements in cluster
        cluster_size = sum(preds == cluster)
        cluster_num_pos = sum(np_labels[preds == cluster])
        cluster_ratio_pos = cluster_num_pos / cluster_size
        cluster_assignment = 1 if cluster_ratio_pos >= 0.5 else 0
        
        # label matches
        label_true = label_match_legend["true pos" if cluster_assignment == 1 else "true neg"]
        label_true_mask = np.logical_and(preds == cluster, np_labels == cluster_assignment)
        label_match[label_true_mask] = label_true
        label_false = label_match_legend["false pos" if cluster_assignment == 1 else "false neg"]
        label_false_mask = np.logical_and(preds == cluster, np_labels != cluster_assignment)
        label_match[label_false_mask] = label_false
        
        # gather element
        label_match_cluster = label_match[preds == cluster]
        cluster_true_pos = label_match_cluster[label_match_cluster == label_match_legend["true pos"]]
        cluster_false_pos = label_match_cluster[label_match_cluster == label_match_legend["false pos"]]
        cluster_true_neg = label_match_cluster[label_match_cluster == label_match_legend["true neg"]]
        cluster_false_neg = label_match_cluster[label_match_cluster == label_match_legend["false neg"]]
            
        if verbose:
            print(f"Cluster {cluster:>2}, num points {cluster_size:>8}, POS labelled {cluster_num_pos:>6}"
                  + f",{round(cluster_ratio_pos,5)*100:>8}%, for {label_name}, on {data_sets}.")

    # assign colors
    point_colors = np.array([label_color_map[i] for i in label_match])
    
    # poltting
    combinations = itertools.combinations(feature_names, 2)
    
    fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    # iterate over combinations for subplots
    for i, combination in enumerate(combinations): 
        df_combo_features = df_features.loc[:, combination]
        np_compo_features = df_combo_features.to_numpy()
        
        # iterate over clusters for labels
        for cluster in clusters_all:
            ax[i //3, i %3].set_xlabel(combination[0])
            ax[i //3, i %3].set_ylabel(combination[1])
            cluster_marker = marker_map[cluster % len(marker_map)]
            ax[i //3, i %3].scatter(np_compo_features[preds == cluster, 0], np_compo_features[preds == cluster, 1],\
                c = point_colors[preds == cluster], marker = cluster_marker)
    fig.tight_layout()
    plt.show()

    