import data_lib
import re
from typing import List
from tkinter_classes import ScrollableInputFrame, ScrollablePlotSelectFrame
from negative_dimnesion_detection import get_negative_dimensions, get_negative_dimension_3
from icecream import ic
import tkinter as tk    
import customtkinter as ctk
from sklearn import cluster
from typing import Dict
import stats_lib
from pathlib import Path
import decision_lib
import plot_lib
import pandas as pd
import numpy.typing as npt
import numpy as np
import transform_lib

class NotNegError(Exception):
    pass

class TkinterSession():
    
    def __init__(self):
        self.files = None
        self.file_masks = None
        self.file_data : pd.DataFrame  = None
        self.neg_files = None
        self.neg_data : pd.DataFrame = None
        self.neg_masks = None
        self.settings_vars : Dict[str, tk.Variable] = {}
        
        self.chamber_file = None

        self.numplots = 6
        self.decision = None
        self.threshold = 0.5
        self.settings_dir = Path(Path.home(),".pcr_conc")
        self.settings_path = Path(self.settings_dir,"settings.csv")
        self.axis_map_path = Path(self.settings_dir,"axis_map.csv")

    def default_settings(self):
        settings_dict_default = { 
            "output_path": str(Path(Path.home(), "pcr_results").absolute()),
            "eps" : 0.3,
            "outliers" : 0.001, 
            "num_plot_points" : 10000,
            "algorithm" : "Hierarchy",
            "negatives_max_positives" : 3,
            "neg_threshold" : 10000,
            "input_path": str(Path(Path.home()).absolute()),
            }

        axis_map_default = {
            "Chan1_FluoValue" : 'SARS-N2',
            "Chan2_FluoValue" : 'SARS-N1',
            "Chan3_FluoValue" : 'IBV-M',
            "Chan4_FluoValue" : 'RSV-N',
            "Chan5_FluoValue" : 'IAV-M',
            "Chan6_FluoValue" : 'MHV',
        }
        return settings_dict_default, axis_map_default
    
    def init_settings(self, master):
        settings_list = [
            tk.DoubleVar(master, None, "eps"),
            tk.IntVar(master, None, "num_plot_points"),
            tk.DoubleVar(master, None, "outliers"),
            tk.StringVar(master, None, "output_path"),
            tk.StringVar(master, None, "algorithm"),
            tk.IntVar(master, None, "negatives_max_positives"),
            tk.DoubleVar(master, None, "neg_threshold"),
            tk.StringVar(master, None, "input_path"),
        ]
        for m in settings_list:
            self.settings_vars[str(m)] = m

    def get_chamber_file(self, file) -> None:

        if self.file_data is None:
            raise Exception("Experiment files must be selected before chamber files.")

        try:
            chamber_file = pd.read_csv(file)
        except:
            raise FileNotFoundError("Error: No file provided.")
        
        chamber_file.rename(columns=lambda x : x.strip(), inplace=True)

        columns = list(chamber_file.columns)
        
        if not "ChamberName" in columns:
            raise Exception("Chamber file does not contain column 'ChamberName'.")

        if not "ChamberID" in columns:
            raise Exception("Chamber file does not contain column 'ChamberID'.")
        
        if not "SampleName" in columns:
            raise Exception("Chamber file does not contain column 'SampleName'.")
        
        _ = self.check_chambers(chamber_file=chamber_file)
        self.chamber_file = chamber_file
        
    def get_files(self, file_list : List[str], axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame) -> None:
        
        file_data, file_masks = data_lib.load_raw_dataset(file_list)

        if not file_data.dtypes.map(pd.api.types.is_numeric_dtype).all():
            raise Exception("Chosen file(s) do have columns with non numeric entries. You have possibly chosen a wrong file.")
        
        self.settings_vars["input_path"].set(Path(file_list[0]).parent.absolute())
        self.store_settings()

        axis_frame.remove_all()
        for col in file_data.columns:
            axis_frame.add_item(col,self.axis_map)
       
        n = file_data.shape[1]
        if n % 2 == 1:
            n = (n+1) // 2
        else:
            n = n // 2
        select_frame.remove_all()
        select_frame.set_labels(file_data.columns.to_list())
        for i in range(n):
            select_frame.add_item(2*i, 2*i + 1)
        
        self.decision = None
        self.file_data, self.file_masks = file_data, file_masks
        self.files = file_list
        
        # this check possibly raises exception so always call it last
        if not self.neg_data is None:
            self.check_negs()

    def get_negs(self, file_list : List[str]) -> None:
        if self.file_data is None:
            raise Exception("Experiment files must be selected before negative control.")
        neg_data, neg_masks = data_lib.load_raw_dataset(file_list)
        
        # this will raise a key error when the columns do not match
        neg_data = neg_data.loc[:, self.file_data.columns]
        
        max_positives = self.settings_vars["negatives_max_positives"].get()
        neg_threshold = self.settings_vars["neg_threshold"].get()
        
        neg_dims = get_negative_dimension_3(np.array(neg_data), max_positives=max_positives, threshold=neg_threshold)

        if np.any(~neg_dims):
            raise NotNegError(f"File list {file_list} contains cluster which are contaminated!")

        self.neg_data, self.neg_masks = neg_data, neg_masks
        self.neg_files = file_list
        
        self.decision = None
    
    def check_negs(self):
        # this will raise a key error when the columns do not match
        try:
            neg_data = neg_data.loc[:, self.file_data.columns]
            self.neg_data = neg_data
        except:
            self.neg_data = None
            self.neg_masks = None
            self.neg_files = None
            raise Exception("New data did not match negative data!")

    def check_chambers(self, chamber_file : pd.DataFrame):
        # will raise an error if the chamber dont match
        if chamber_file is None:
            return None
        file_info = {}
        for file in self.files:
            file_repl = file.replace("-","_")
            for i,ch_id in enumerate(chamber_file.loc[:,"ChamberID"].to_list()):
                a = ch_id.strip().replace("-","_")
                x = re.search(a,file_repl)
                if not x is None:
                    file_info[file] = i
            if not file in file_info.keys():
                raise Exception("Chamber does not contain chamber data of all selected files.")
        return file_info
            
    def drop_negs(self):
        self.neg_data = None
        self.neg_files = None
        self.neg_masks = None 
        self.decision = None
        
    def get_channel_names(self, axis_frame : ScrollableInputFrame) -> List[str]:
        if self.file_data is None:
            raise Exception("No Experiment files are selected yet.")
        axis_labels : List[ctk.CTkEntry] = axis_frame.input_list
        axis_labels : List[tk.StringVar] = [ e.cget("textvariable")  for e in axis_labels]

        dims = self.file_data.shape[1]
        label_list = [""]*dims

        for a in axis_labels:
            label = a.get().strip()
            axis = str(a)
            index = self.file_data.columns.get_loc(axis)
            label_list[index] = label
        
        return label_list, axis_labels

    def get_plot_selections(self, axis_frames : ScrollablePlotSelectFrame, axis_labels : List[str]):
        if self.file_data is None:
            raise Exception("No Experiment files are selected yet.")
        left_list : List[ctk.CTkComboBox] = axis_frames.left_list
        right_list : List[ctk.CTkComboBox] = axis_frames.right_list
        left_list : List[tk.StringVar] = [ e.cget("variable")  for e in left_list]
        right_list : List[tk.StringVar] = [ e.cget("variable")  for e in right_list]
        
        rows = len(right_list)
        label_list = [""]*(2 * rows)

        for i in range(rows) :
            r = right_list[i]
            l = left_list[i]
            r = r.get().strip()
            l = l.get().strip()
            r_index = self.file_data.columns.get_loc(r)
            l_index = self.file_data.columns.get_loc(l)
            label_list[2*i] = (axis_labels[l_index], axis_labels[r_index])
            label_list[2*i+1] = (axis_labels[r_index], axis_labels[l_index])
        
        return label_list
        
    def compute_clusters(self):
        
        np_data = self.file_data.to_numpy()
        if self.neg_data is None:
            np_neg = None
        else:
            np_neg = self.neg_data.to_numpy()

        whitening_engine = transform_lib.WhitenTransformer(whiten=transform_lib.Whitenings.NONE)
        num_cluster = int(2**np_data.shape[1] * 1.2)
        num_cluster = min(np_data.shape[0], num_cluster)
        cluster_engine = cluster.KMeans(n_clusters=num_cluster, n_init='auto')
        
        outlier_quantile = self.settings_vars["outliers"].get()
        if outlier_quantile <= 0:
            outlier_quantile = 1e-15
            self.settings_vars["outliers"].set(outlier_quantile)
        elif outlier_quantile > 0.5:
            outlier_quantile = 0.5
            self.settings_vars["outliers"].set(outlier_quantile)
            
            
        max_positives = self.settings_vars["negatives_max_positives"].get()
        if max_positives < 0:
            max_positives = 0
            self.settings_vars["negatives_max_positives"].set(max_positives)
        
        threshold = self.settings_vars["neg_threshold"].get()
        if threshold < 0:
            threshold = 0
            self.settings_vars["neg_threshold"].set(threshold)

        if self.settings_vars["algorithm"].get() == "Hierarchy":
            self.decision = decision_lib.ClusterRelativeHierarchyMeanClassifier(
                                                cluster_algorithm=cluster_engine,
                                                whitening_transformer=whitening_engine,
                                                contamination=outlier_quantile,
                                                negative_range=None,
                                                )
        else:
            self.decision = decision_lib.WhitnesDensityClassifier(
                                                cluster_algorithm=cluster_engine,
                                                whitening_transformer=whitening_engine,
                                                outlier_quantile=outlier_quantile,
                                                max_positive=max_positives,
                                                negative_theshold=threshold,
                                                negative_range=None,
                                                verbose=True,
                                                )

        self.decision.read_data(np_data,None,None)
    
    def compute(self, axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame):
        
        if self.decision is None:
            self.compute_clusters()
            self.settings_vars["num_plot_points"].set(10000)
        
        self.decision.eps = self.settings_vars["eps"].get()
        if self.settings_vars["algorithm"].get() == "Hierarchy":
            self.decision.eps = None

        self.decision.prediction_axis, self.axis_labels = self.get_channel_names(axis_frame=axis_frame)
        
        self.decision.predict_all()
        df_data_points = pd.DataFrame(data=self.decision.X, columns=self.decision.prediction_axis) 
        df_predictions = self.decision.probabilities_df
        selected_pairs = self.get_plot_selections(axis_frames=select_frame, axis_labels=self.decision.prediction_axis)

        n = self.decision.X.shape[0]
        p = np.clip(self.settings_vars["num_plot_points"].get() / n,0,1)
        mask = np.random.choice([0,1], n, replace=True, p=[1-p,p] ).astype(bool)
        fig = plot_lib.plot_pairwise_selection_bayesian_no_gt(df_data_points,df_predictions,selected_pairs,n_cols=2,mask=mask)
        
        short_res = stats_lib.compute_short_results(self.decision.probabilities_df, self.threshold, df_data_points)
        
        return fig, short_res
        
    def export(self, axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame):

        chamber_map = self.check_chambers(chamber_file=self.chamber_file)
        
        if self.decision is None:
            self.compute_clusters()
            self.settings_vars["num_plot_points"].set(10000)
        
        if self.decision.probabilities_df is None:
            _ = self.compute(axis_frame=axis_frame,select_frame=select_frame)
        
        # compute path
        output_dir = Path(self.settings_vars["output_path"].get())
        output_dir.mkdir(parents=True, exist_ok=True) # create directory if it does not yet exist
        result_path = Path(output_dir, "results.csv")

        df_list = []
        df_data_points = pd.DataFrame(data=self.decision.X, columns=self.decision.prediction_axis) 
        df_data_points_chan = pd.DataFrame(data=self.decision.X, columns=self.file_data.columns) 
        df_probabilities = pd.DataFrame(data=self.decision.probabilities_df, columns=self.decision.prediction_axis) 
        for file in self.files:
            file_path = Path(file)
            test_name = file_path.stem
            mask = self.file_masks[file]
            df_temp = stats_lib.compute_results(self.decision.probabilities_df.iloc[mask,:],
                                                     self.threshold,
                                                     df_data_points.iloc[mask,:])
            if chamber_map is None:
                df_temp.insert(0, "Chamber", [test_name])
                data_name = test_name
            else:
                def ins(col):
                    index = chamber_map[file]
                    val = self.chamber_file.loc[index, col]
                    df_temp.insert(0, col, val)
                ins("PoolingID")
                ins("ChamberContext")
                ins("SampleName")
                ins("ChamberID")
                ins("ChamberName")
                data_name = self.chamber_file.loc[chamber_map[file], "ChamberID"]
            df_list.append(df_temp)
            
            # ouput results
            file_path = Path(output_dir, f"{data_name.strip()}_labelled.csv")
            df_output = pd.concat([ df_data_points_chan.iloc[mask,:], df_probabilities.iloc[mask,:].astype("Float32")],axis=1, join="outer")
            df_output.to_csv(file_path)
            
        # compute plots
        selected_pairs = self.get_plot_selections(axis_frames=select_frame, axis_labels=self.decision.prediction_axis)
        for (col_one, col_two) in selected_pairs:
            plot_path = Path(output_dir, f"{col_one}_labels.png")
            plot_lib.plot_pair_bayesian_no_gt(df_data_points, df_probabilities, col_one, col_two, save_path=plot_path)

            

        df_results = pd.concat(df_list)
        df_results.to_csv(result_path, index=False)
        
    
    def load_settings(self, change_config : bool = True):
        
        settings_dict_default, axis_map_default = self.default_settings()

        settings = {}
        if self.settings_path.is_file():
            settings = pd.read_csv(self.settings_path).to_dict(index="False")
            
        # pd.Dataframe.to_dict() returns dictionairies for every key! Has to be transformed back
        for k in settings.keys():
            settings[k] = settings[k][0]
        settings = settings_dict_default | settings

        axis_map = {}
        if self.axis_map_path.is_file():
            axis_map = pd.read_csv(self.axis_map_path).to_dict(index="False")
        
        for k in axis_map.keys():
            axis_map[k] = axis_map[k][0]
        axis_map = axis_map_default | axis_map
        
        if change_config:
            for s in self.settings_vars.keys():
                if s in settings.keys():
                    self.settings_vars[s].set(settings[s])
            self.axis_map = axis_map
        
        return axis_map, settings
    
    def settings_dict(self):
        settings_dict = {}
        for s in self.settings_vars.keys():
            settings_dict[s] = self.settings_vars[s].get()
        return settings_dict
            
            
    def store_settings(self, settings=True, axis=True, key = None):
        
        # first get settings
        stored_axis_map, stored_settings = self.load_settings(change_config=False)
        
        # comine with local settings not to loose anything
        if key is None:
            axis_map = stored_axis_map | self.axis_map
            settings = stored_settings | self.settings_dict()
        else:
            axis_map = stored_axis_map 
            settings = stored_settings
            settings[key] = self.settings_dict()[key]
        
        # convert values to strings or numbers instead of tkinter variables
        df_settings = pd.DataFrame(settings, index=[0])
        df_axis_map = pd.DataFrame(axis_map, index=[0])
        
        # create settings path if it does not exists
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        
        # store settings
        if settings:
            df_settings.to_csv(self.settings_path,index=False)
        if axis:
            df_axis_map.to_csv(self.axis_map_path,index=False)
