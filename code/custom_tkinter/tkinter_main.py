import customtkinter as ctk
from CTkMessagebox import CTkMessagebox as msg
from tkinter import filedialog as fd
import tkinter as tk
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter_classes import ScrollableInputFrame, ScrollablePlotSelectFrame
from tkinter_backend import TkinterSession
from CTkTable import CTkTable
from pathlib import Path
import pandas as pd
import time

class ctkApp:
        
    def __init__(self):

        ctk.set_appearance_mode("system")
        self.root = ctk.CTk()
        self.root.geometry("1200x800+200x200")
        self.root.title("PCR Contamination")
        self.root.update()

        self.session = TkinterSession()
        self.session.init_settings(self.root)
        self.session.load_settings()

        self.font = (tk.font.nametofont("TkDefaultFont"), 15)
        self.titlefont = (tk.font.nametofont("TkDefaultFont"), 20)
        bg_color = self.root.cget("fg_color")

        self.root.grid_columnconfigure(0,weight=1,uniform=1)
        self.root.grid_rowconfigure(0,weight=1)

        self.tab_view = ctk.CTkTabview(self.root)
        self.tab_view.grid(row=0,column=0,padx=(0,0),pady=(0,0), sticky="news" )
    

        self.tab_plots = self.tab_view.add("Experiments")
        self.tab_plots.grid_columnconfigure(0,minsize=400,weight=1,uniform=1)
        self.tab_plots.grid_columnconfigure(1,weight=10,uniform=1)
        self.tab_plots.grid_rowconfigure(0,weight=10)
        self.tab_plots.grid_rowconfigure(1,weight=2)

        self.tab_settings = self.tab_view.add("Settings")
        
        ###############################
        # Experiments
        ###############################

        self.plot_frame = ctk.CTkFrame(master=self.tab_plots)
        self.plot_frame.grid(column=1,row=0, sticky="snwe",pady=(5,0), padx=(5,5))
        self.plot_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.plot_frame.grid_rowconfigure(0,weight=1,uniform=1)

        self.res_frame = ctk.CTkFrame(master=self.tab_plots)
        self.res_frame.grid(column=1,row=1, sticky="we",pady=(5,5), padx=(5,5))
        self.res_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.res_frame.grid_rowconfigure(1,weight=1)
        self.res_frame.grid_rowconfigure(0,weight=1)

        self.control_frame = ctk.CTkFrame(master=self.tab_plots,
                                  fg_color=bg_color,
                                  )
        self.control_frame.grid(column=0,row=0,rowspan=2, sticky="news",pady=(5,5), padx=(5,5))
        self.control_frame.grid_columnconfigure(0,weight=25)
        self.control_frame.grid_columnconfigure(1,weight=15)

        button_pad = 10
        pad_x = 20
        pad_x_inter = 10
        pad_y_inter = 3
        button_height = self.font[1] + 15
        rowcnt = 0

        self.experiment_title = ctk.CTkLabel(self.control_frame, text="Experiment Setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.experiment_title.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")
        rowcnt = rowcnt+1

        self.files_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Experiment files",
                               font=self.font,
                               height=button_height,
                               command=self.select_files)
        self.files_button.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad,0), sticky="news")
        self.orig_button_color = self.files_button.cget("fg_color")
        rowcnt = rowcnt+1

        self.neg_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Negative control",
                               height=button_height,
                               font=self.font,
                               command=self.select_neg)
        self.neg_button.grid(row=rowcnt,column=0,padx=(pad_x,pad_x_inter),pady=(button_pad,0), sticky="news" )

        self.neg_button_cancel = ctk.CTkButton(master = self.control_frame,
                               text="Unselect",
                               height=button_height,
                               font=self.font,
                               command=self.cancel_neg)
        self.neg_button_cancel.grid(row=rowcnt,column=1,padx=(pad_x_inter,pad_x),pady=(button_pad,0), stick="news")
        rowcnt = rowcnt+1

        self.cluster_buttton = ctk.CTkButton(master = self.control_frame,
                               text="Compute Clusters",
                               font=self.font,
                               height=button_height,
                               command=self.compute_clusters)
        self.cluster_buttton.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.axis_name_title = ctk.CTkLabel(self.control_frame, text="Choose Axis Names:",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.axis_name_title.grid(row=rowcnt, column=0, columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.axis_name_frame = ScrollableInputFrame(master=self.control_frame,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    )
        self.axis_name_frame.grid(row=rowcnt, column=0, columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.slider_label = ctk.CTkLabel(self.control_frame, text="Choose relative distance eps",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.slider_label.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.slider = ctk.CTkSlider(master=self.control_frame, from_=0, to=1, number_of_steps=50, variable=self.session.settings_vars["eps"])
        self.slider.grid(row=rowcnt, column=0, padx=(pad_x,0), pady=(button_pad,0), sticky="news")
        self.slider_display = ctk.CTkEntry(master=self.control_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["eps"],
                                    justify='center',
                            )
        self.slider_display.grid(row=rowcnt, column=1, padx=(pad_x_inter,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.plot_title = ctk.CTkLabel(self.control_frame, text="Plot setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(pad_y_inter,0), sticky="news")
        rowcnt = rowcnt+1

        self.plot_selection = ScrollablePlotSelectFrame(master=self.control_frame,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    )
        self.plot_selection.grid(row=rowcnt, column=0, columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="ew")
        rowcnt = rowcnt+1

        self.plot_point_slider_label = ctk.CTkLabel(self.control_frame, text="Number of points to be plotted",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.plot_point_slider_label.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.plot_point_slider = ctk.CTkSlider(master=self.control_frame,
                                               from_=0,
                                               to=1,
                                               number_of_steps=50,
                                               variable=self.session.settings_vars["num_plot_points"])
        self.plot_point_slider.grid(row=rowcnt, column=0, padx=(pad_x,0), pady=(button_pad,0), sticky="news")
        self.plot_point_slider_display = ctk.CTkEntry(master=self.control_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["num_plot_points"],
                                    justify='center',
                            )
        self.plot_point_slider_display.grid(row=rowcnt, column=1, padx=(pad_x_inter,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.compute_button = ctk.CTkButton(master = self.control_frame,
                               text="Compute and Plot",
                               height=button_height,
                               font=self.font,
                               command=self.compute)
        self.compute_button.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.compute_button = ctk.CTkButton(master = self.control_frame,
                               text="Export Results and Plots",
                               height=button_height,
                               font=self.font,
                               command=self.export)
        self.compute_button.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1
        
        # configure results
        self.plot_title = ctk.CTkLabel(self.res_frame, text="Results",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=0, column=0, padx=(pad_x,pad_x), pady=(pad_y_inter,pad_y_inter), sticky="news")
        
        #############################
        # End Experiments
        #############################

        #############################
        # Settings
        #############################
        self.settings_frame = ctk.CTkFrame(master=self.tab_settings)
        self.settings_frame.grid(column=0,row=0, sticky="snwe",pady=(5,0), padx=(5,5))
        self.settings_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.settings_frame.grid_rowconfigure(0,weight=1,uniform=1)

        rowcnt=0
        tmp_title = ctk.CTkLabel(self.settings_frame, text="General computation",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_textbox = ctk.CTkTextbox(self.settings_frame,
                             font=self.font,
                             wrap='word',
                             )
        tmp_textbox.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")
        tmp_textbox.insert(0.0,text=r"""
                            Generally we work under the following assumptions:
                            1. Let D be the set of dimensions. If for a sample there exists points
                            which are positive in $S \subseteq D$. Then we assume that for every
                            $S' \subset S$ there exists a 'cluster' of points only positive in $S'$.

                            2. Based on based on previous assumption, let $c$ be a 'cluster' of points
                            positive in $S \subseteq D$ and $c'$ positive in $S' \subset S$, then
                            we have that the mean of $c$ is larger equal the mean of $c'$ in all
                            dimensions (this expresses that we expect only positive correlations)

                            3. Samples for which the values in one dimension do not exceed the
                            absolute value 10000 after outlier removal are negative in this dimension.
                            """
                           )
        rowcnt = rowcnt+1
        
        self.outliers = ctk.CTkLabel(self.settings_frame, text="Outlier percentile (amount of outlier points to be removed)",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.outliers.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.outliers_slider = ctk.CTkSlider(master=self.settings_frame, from_=0, to=0.1, number_of_steps=500, variable=self.session.settings_vars["outliers"])
        self.outliers_slider.grid(row=rowcnt, column=0, padx=(pad_x,0), pady=(button_pad,0), sticky="news")
        self.outliers_slider_display = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["outliers"],
                                    justify='center',
                            )
        self.outliers_slider_display.grid(row=rowcnt, column=1, padx=(pad_x_inter,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_label = ctk.CTkLabel(self.settings_frame, wraplength=500, text="Negative threshhold (points which are below that number times "
                                 + "the max of the zero cluster in all dimensions are ignored (only used for speedup))",
                             justify='center',
                             font=self.font,
                             anchor="center")
        tmp_label.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_slider = ctk.CTkSlider(master=self.settings_frame, from_=0, to=1, number_of_steps=500, variable=self.session.settings_vars["neg_ignore"])
        tmp_slider.grid(row=rowcnt, column=0, padx=(pad_x,0), pady=(button_pad,0), sticky="news")
        tmp_display = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["neg_ignore"],
                                    justify='center',
                            )
        tmp_display.grid(row=rowcnt, column=1, padx=(pad_x_inter,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1


        tmp_title = ctk.CTkLabel(self.settings_frame, text="Negative control detection",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_textbox = ctk.CTkTextbox(self.settings_frame,
                             font=self.font,
                             wrap='word',
                             )
        tmp_textbox.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")
        tmp_textbox.insert(0.0,text="""
                            This works under the assumption that points are normally distributed
                            in dimension in which all are negative and not normally distributed in dimensions,
                            where all some positive.
                            
                            Under this assumption we make a normaltest
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
                            
                            According to experiments performed on each chamber on the given data available for coding,
                            we found that dimensions in which we found only a negative cluster the test statistic takes
                            in average and a standard deviation (over all samples and dimensions
                            in which a dimension only contains a negative cluster) of
                            mean: 1.54
                            std: 1.26
                            As opposed to samples and dimensions in which there exist positive droples:
                            mean: 0.077
                            std: 0.088
                            On negative contols we even have
                            mean: 0.036
                            std: 0.034
                            
                            By this clear distinction, it is reasonable to distinguish based on this statistic.
                            We consider samples (dimension in one sample) to have a probability of 0.99
                            that they correspond to the given class (all negative, some positive).
                            Then we further assume that in the middle of the two means we have a
                            probability of 0.5. Using this, we then fit a sigmoid and return the result
                            for a new cluster.
                            """
                           )
        rowcnt = rowcnt+1

        tmp_label = ctk.CTkLabel(self.settings_frame, text="max contamination (the maximal point of points to be positive in any dimension `not that this is unrealisitc`).",
                             justify='center',
                             font=self.font,
                             anchor="center")
        tmp_label.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.nc_outliers = ctk.CTkLabel(self.settings_frame, text="What is the amount of outliers not to be considered in the negative control detection",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.nc_outliers.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.nc_outliers_slider = ctk.CTkSlider(master=self.settings_frame, from_=0, to=0.01, number_of_steps=500, variable=self.session.settings_vars["nc_outliers"])
        self.nc_outliers_slider.grid(row=rowcnt, column=0, padx=(pad_x,0), pady=(button_pad,0), sticky="news")
        self.nc_outliers_slider_display = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["nc_outliers"],
                                    justify='center',
                            )
        self.nc_outliers_slider_display.grid(row=rowcnt, column=1, padx=(pad_x_inter,pad_x), pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_title = ctk.CTkLabel(self.settings_frame, text="Export Settings",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(5,0), sticky="news")

        
        rowcnt = rowcnt+1
        button = ctk.CTkButton(master = self.settings_frame,
                               text="export directory (exported results will be stored there)",
                               height=button_height,
                               font=self.font,
                               command=self.choose_output_dir)
        button.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        label = ctk.CTkLabel(self.settings_frame, textvariable=self.session.settings_vars["output_path"],
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0,columnspan=2, padx=(pad_x,pad_x), pady=(pad_y_inter,0), sticky="news")
        rowcnt = rowcnt+1

        button = ctk.CTkButton(master = self.settings_frame,
                               text="Save as default",
                               height=button_height,
                               font=self.font,
                               command=self.save_default)
        button.grid(row=rowcnt,column=0, columnspan=2,padx=(pad_x,pad_x),pady=(button_pad + 20,0), sticky="news")
        rowcnt = rowcnt+1
        
        
        #########################
        # END SETTINGS
        ########################

        self.root.protocol("WM_DELETE_WINDOW", self.destroy_root)

        self.root.mainloop()
   
   
   
   
    
    def destroy_root(self):
        self.root.quit()
        
    def select_files(self):
        try:
            paths = fd.askopenfilenames(parent=self.root, filetypes=[("csv files", "*.csv")], title="Choose files to be processed", initialdir="/home/nico/csem/lab/Data/6P-positive-dilution-series-1-unlabelled/droplet-level-data/RawData")
            self.session.get_files(paths, axis_frame=self.axis_name_frame, select_frame=self.plot_selection)
            self.files_button.configure(require_redraw=True, fg_color="green")
            if self.session.neg_data is None:
                self.neg_button.configure(require_redraw=True, fg_color=self.orig_button_color)
            if self.session.decision is None:
                self.cluster_buttton.configure(require_redraw=True, fg_color=self.orig_button_color)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def choose_output_dir(self):
        try:
            out_dir = fd.askdirectory(parent=self.root, title="Output directory", initialdir=Path.home())
            self.session.settings_vars["output_path"].set(out_dir)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def compute_clusters(self):
        try:
            self.session.compute_clusters()
            self.cluster_buttton.configure(require_redraw=True, fg_color="green")
            self.plot_point_slider._to = self.session.decision.X_transformed.shape[0]
            self.session.settings_vars["num_plot_points"].set(10000)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def select_neg(self):
        try:
            paths = fd.askopenfilenames(parent=self.root, filetypes=[("csv files", "*.csv")], title="Select Negative Controls", initialdir="/home/nico/csem/lab/Data/6P-positive-dilution-series-1-unlabelled/droplet-level-data/RawData")
            self.session.get_negs(paths)
            self.neg_button.configure(require_redraw=True, fg_color="green")
            if self.session.decision is None:
                self.cluster_buttton.configure(require_redraw=True, fg_color=self.orig_button_color)
            msg(title="Valid", message="Valid negative control.", icon="check")
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def cancel_neg(self):
        try:
            self.session.drop_negs()
            self.neg_button.configure(require_redraw=True, fg_color=self.orig_button_color)
            if self.session.decision is None:
                self.cluster_buttton.configure(require_redraw=True, fg_color=self.orig_button_color)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def compute(self):
        #try:
            fig, df_results = self.session.compute(self.axis_name_frame, self.plot_selection)
            self.cluster_buttton.configure(require_redraw=True, fg_color="green")
            self.plot_point_slider._to = self.session.decision.X_transformed.shape[0]
            self.session.store_settings(axis=True, settings=False)
            self.draw_results(df_results)
            self.draw_figure(fig)
        #except Exception as e:
        #    msg(title="Error", message=str(e), icon="cancel")

    def export(self):
        #try:
            self.session.export(self.axis_name_frame, self.plot_selection)
            self.session.store_settings(axis=True, settings=False)
            msg(title="Export", message="Successful export.", icon="check")
        #except Exception as e:
        #    msg(title="Error", message=str(e), icon="cancel")
        
    def save_default(self):
        #try:
            self.session.store_settings(axis=False)
            msg(title="Export", message="Successfully saved settings as default.", icon="check")
        #except Exception as e:
        #    msg(title="Error", message=str(e), icon="cancel")
        
    def draw_results(self, df_results : pd.DataFrame):
        shape = df_results.shape
        data = [df_results.columns.to_list()]
        for i in range(shape[0]):
            data.append(df_results.iloc[i,:].to_list())
        table = CTkTable(self.res_frame, values=data, row=(shape[0] + 1), column=shape[1])
        table.grid(row=1, column=0, padx=(5,5), pady=(5,5), sticky="nswe")

    def draw_figure(self, fig):

        canvas = FigureCanvasTkAgg(fig,master=self.plot_frame)
        canvas.get_tk_widget().place(relx=0, rely=0, relheight=1,relwidth=1)
        canvas.draw()
        self.root.update()
    
        
if __name__ == "__main__":        
    CTK_Window = ctkApp()