import pandas as pd
from icecream import ic
from typing import Dict, List, Tuple
import numpy.typing as npt
import numpy as np
import re
import pathlib
import itertools
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
            for a,b,c in itertools.zip_longest(group_elems[::3], group_elems[1::3], group_elems[2::3], fillvalue=""):
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

def load_raw_dataset(files: List[str]) -> Tuple[pd.DataFrame, Dict[str, npt.NDArray]]:
    """Generate Dataframe with all the features and a mask for each individual file stored in a file with the correpsonding name

    Args:
        files (List[str]): List of Lists of filenames
    
    Exceptions:
        FileNotFoundError: For given files not ending with .cvs or path related problems
        KeyError: For files not containing the right colums.

    Returns:
        Tuple[pd.DataFrame, Dict[str, npt.NDArray]: Dataframe with one row for each point found in one of the files, Dictionarray containing files and corresponding masks
    """
    if len(files) <= 0:
        raise FileNotFoundError("Error: No files provided.")
    
    df_list : List[pd.DataFrame] = []
    for file in files:
        
        path = pathlib.Path(file)
        
        if not path.is_file():
            raise FileNotFoundError(f"Error: {path} is not a file.")
        
        if not path.suffix == ".csv":
            raise FileNotFoundError(f"Error: {path} is not a csv file.")
        
        col_pattern = "^Chan.*"
        raw_data = pd.read_csv(str(path))
        raw_data.rename(columns=lambda x : x.strip(), inplace=True)
        
        keep_cols = list(filter(re.compile(col_pattern).findall, raw_data.columns))
        
        df_list.append(raw_data.loc[:,keep_cols])
        
    # complete major df
    cols = df_list[0].columns
    
    # reorder columns of all dataframes
    start_indices = [0]
    for i,df in enumerate(df_list):
        try:
            df = df.loc[:, cols]
        except:
            raise KeyError(f"File {files[i]} does not contain the needed columns: {cols.to_list}." +
                           f"\n\nNote that we assume {files[0]} contains the right colums and we only consider columns which names start with 'Chan'")
        start_indices.append(start_indices[len(start_indices)-1] + df.shape[0])
    
    df_major = pd.concat(df_list, axis=0)
    file_masks : Dict[str, npt.NDArray] = {}
    for i, file in enumerate(files):
        mask = np.zeros(df_major.shape[0],dtype=bool)
        mask[start_indices[i]:start_indices[i+1]] = True
        file_masks[file] = mask
    
    return df_major, file_masks