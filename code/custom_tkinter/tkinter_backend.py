import data_lib
from typing import List
from tkinter_classes import ScrollableInputFrame, ScrollablePlotSelectFrame
from negative_dimnesion_detection import get_negative_dimensions
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

        self.numplots = 6
        self.decision = None
        self.min_threshold = 0.1
        self.max_threshold = 0.9
        self.settings_dir = Path(Path.home(),".pcr_conc")
        self.settings_path = Path(self.settings_dir,"settings.csv")
        self.axis_map_path = Path(self.settings_dir,"axis_map.csv")

    def default_settings(self):
        settings_dict_default = { 
            "output_path": str(Path(Path.home(), "pcr_results").absolute()),
            "eps" : 0.3,
            "outliers" : 0.001, 
            "nc_outliers" : 0.01,
            "neg_ignore" : 0.9,
            "num_plot_points" : 10000,
            }

        axis_map_default = {
            "Chan1_FluoValue" : 'SARS-N2_POS',
            "Chan2_FluoValue" : 'SARS-N1_POS',
            "Chan3_FluoValue" : 'IBV-M_POS',
            "Chan4_FluoValue" : 'RSV-N_POS',
            "Chan5_FluoValue" : 'IAV-M_POS',
            "Chan6_FluoValue" : 'MHV_POS',
        }
        return settings_dict_default, axis_map_default
    
    def init_settings(self, master):
        settings_list = [
            tk.DoubleVar(master, None, "eps"),
            tk.IntVar(master, None, "num_plot_points"),
            tk.DoubleVar(master, None, "outliers"),
            tk.DoubleVar(master, None, "nc_outliers"),
            tk.DoubleVar(master, None, "neg_ignore"),
            tk.StringVar(master, None, "output_path")
        ]
        for m in settings_list:
            self.settings_vars[str(m)] = m
    
    def get_files(self, file_list : List[str], axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame) -> None:
        
        file_data, file_masks = data_lib.load_raw_dataset(file_list)

        if not file_data.dtypes.map(pd.api.types.is_numeric_dtype).all():
            raise Exception("Chosen file(s) do have columns with non numeric entries. You have possibly chosen a wrong file.")

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
            self.ckeck_negs()

    def get_negs(self, file_list : List[str]) -> None:
        if self.file_data is None:
            raise Exception("Experiment files must be selected before negative control.")
        neg_data, neg_masks = data_lib.load_raw_dataset(file_list)
        
        # this will raise a key error when the columns do not match
        neg_data = neg_data.loc[:, self.file_data.columns]
        
        acceptable_contamination = self.settings_vars["nc_outliers"].get()
        
        neg_dims, _ = get_negative_dimensions(neg_data, outliers_percentile=acceptable_contamination)
        neg_dims = neg_dims <= 0.01 

        if np.any(~neg_dims):
            raise NotNegError(f"File list {file_list} contains cluster which are contaminated!")

        self.neg_data, self.neg_masks = neg_data, neg_masks
        self.neg_files = file_list
        
        self.decision = None
    
    def ckeck_negs(self):
        # this will raise a key error when the columns do not match
        try:
            neg_data = neg_data.loc[:, self.file_data.columns]
            self.neg_data = neg_data
        except:
            self.neg_data = None
            self.neg_masks = None
            self.neg_files = None
            raise Exception("New data did not match negative data!")
            
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
        
        return label_list

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
        num_cluster = int(2**np_data.shape[1] * 2)
        num_cluster = min(np_data.shape[0], num_cluster)
        cluster_engine = cluster.KMeans(n_clusters=num_cluster, n_init='auto')
        
        outlier_quantile = self.settings_vars["outliers"].get()
        negative_range = self.settings_vars["neg_ignore"].get()

        self.decision = decision_lib.WhitnesDensityClassifier(
                                              cluster_algorithm=cluster_engine,
                                              whitening_transformer=whitening_engine,
                                              outlier_quantile=outlier_quantile,
                                              verbose=True,
                                              )

        self.decision.read_data(np_data,np_neg,negative_range)
    
    def compute(self, axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame):
        
        if self.decision is None:
            self.compute_clusters()
            self.settings_vars["num_plot_points"].set(10000)
        
        self.decision.eps = self.settings_vars["eps"].get()
        self.decision.prediction_axis = self.get_channel_names(axis_frame=axis_frame)
        
        self.decision.predict_all()
        df_data_points = pd.DataFrame(data=self.decision.X_transformed, columns=self.decision.prediction_axis) 
        df_predictions = self.decision.probabilities_df
        selected_pairs = self.get_plot_selections(axis_frames=select_frame, axis_labels=self.decision.prediction_axis)

        n = self.decision.X.shape[0]
        p = np.clip(self.settings_vars["num_plot_points"].get() / n,0,1)
        mask = np.random.choice([0,1], n, replace=True, p=[1-p,p] ).astype(bool)
        fig = plot_lib.plot_pairwise_selection_bayesian_no_gt(df_data_points,df_predictions,selected_pairs,n_cols=2,mask=mask)
        
        short_res = stats_lib.compute_short_results(self.decision.probabilities_df, self.min_threshold, self.max_threshold, df_data_points)
        
        return fig, short_res
        
    def export(self, axis_frame : ScrollableInputFrame, select_frame : ScrollablePlotSelectFrame):

        if self.decision is None:
            self.compute_clusters()
            self.settings_vars["num_plot_points"].set(10000)
        
        if self.decision.probabilities_df is None:
            _ = self.compute(axis_frame=axis_frame,select_frame=select_frame)

        df_list = []
        for file in self.files:
            file_path = Path(file)
            test_name = file_path.stem
            mask = self.file_masks[file]
            df_data_points = pd.DataFrame(data=self.decision.X_transformed, columns=self.decision.prediction_axis) 
            df_temp = stats_lib.compute_results(self.decision.probabilities_df.iloc[mask,:],
                                                     self.min_threshold,
                                                     self.max_threshold,
                                                     df_data_points.iloc[mask,:])
            df_temp.loc[:,"Chamber"] = [test_name]
            df_list.append(df_temp)

        df_results = pd.concat(df_list)
        chamber = df_results.pop('Chamber')
        df_results.insert(0, 'Chamber', chamber)
        output_path = Path(self.settings_vars["output_path"].get(), "results.csv")
        df_results.to_csv(output_path, index=False)
        
    
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
            
            
    def store_settings(self, settings=True, axis=True):
        
        # first get settings
        stored_axis_map, stored_settings = self.load_settings(change_config=False)
        
        
        # comine with local settings not to loose anything
        axis_map = stored_axis_map | self.axis_map
        settings = stored_settings | self.settings_dict()
        
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
