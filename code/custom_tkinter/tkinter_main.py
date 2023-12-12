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
        ctk.set_widget_scaling(0.7) # fix windows
        ctk.set_widget_scaling(1.0)
        self.root = ctk.CTk()
        self.root.geometry("1200x800+200x200")
        self.root.title("PCR Contamination")
        self.root.update()

        self.session = TkinterSession()
        self.session.init_settings(self.root)
        self.session.load_settings()

        self.font = (tk.font.nametofont("TkDefaultFont"), 13)
        self.titlefont = (tk.font.nametofont("TkDefaultFont"), 18)
        bg_color = self.root.cget("fg_color")

        self.root.grid_columnconfigure(0,weight=1,uniform=1)
        self.root.grid_rowconfigure(0,weight=1)

        self.tab_view = ctk.CTkTabview(self.root)
        self.tab_view.grid(row=0,column=0,padx=(0,0),pady=(0,0), sticky="news" )
    

        self.tab_plots = self.tab_view.add("Experiments")
        self.tab_plots.grid_columnconfigure(0,minsize=400,weight=1,uniform=1)
        self.tab_plots.grid_columnconfigure(1,weight=3,uniform=1)
        self.tab_plots.grid_rowconfigure(0,weight=10)
        self.tab_plots.grid_rowconfigure(1,weight=2)

        self.tab_settings = self.tab_view.add("Settings")
        self.tab_settings.grid_columnconfigure(0,weight=1,uniform=1)
        self.tab_settings.grid_rowconfigure(0,weight=1,uniform=1)
        
        ###############################
        # Experiments
        ###############################

        self.plot_frame = ctk.CTkFrame(master=self.tab_plots)
        self.plot_frame.grid(column=1,row=0, sticky="snwe",pady=(0,0), padx=(5,5))
        self.plot_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.plot_frame.grid_rowconfigure(0,weight=1,uniform=1)

        self.res_frame = ctk.CTkFrame(master=self.tab_plots)
        self.res_frame.grid(column=1,row=1, sticky="we",pady=(5,5), padx=(5,5))
        self.res_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.res_frame.grid_rowconfigure(1,weight=1)
        self.res_frame.grid_rowconfigure(0,weight=1)

        self.control_frame = ctk.CTkScrollableFrame(master=self.tab_plots,
                                  fg_color=bg_color,
                                  )
        self.control_frame.grid(column=0,row=0,rowspan=2, sticky="news",pady=(0,5), padx=(5,5))
        self.control_frame.grid_columnconfigure(0,weight=25)
        self.control_frame.grid_columnconfigure(1,weight=15)

        self.button_pad = 8
        self.pad_x = 20
        self.pad_x_inter = 8
        self.pad_y_inter = 2
        button_height = self.font[1] + 12
        rowcnt = 0

        self.experiment_title = ctk.CTkLabel(self.control_frame, text="Experiment Setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.experiment_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="news")
        rowcnt = rowcnt+1

        self.files_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Experiment files",
                               font=self.font,
                               height=button_height,
                               command=self.select_files)
        self.files_button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        self.orig_button_color = self.files_button.cget("fg_color")
        rowcnt = rowcnt+1

        self.neg_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Negative control",
                               height=button_height,
                               font=self.font,
                               command=self.select_neg)
        self.neg_button.grid(row=rowcnt,column=0,padx=(self.pad_x,self.pad_x_inter),pady=(self.button_pad,0), sticky="news" )

        self.neg_button_cancel = ctk.CTkButton(master = self.control_frame,
                               text="Unselect",
                               height=button_height,
                               font=self.font,
                               command=self.cancel_neg)
        self.neg_button_cancel.grid(row=rowcnt,column=1,padx=(self.pad_x_inter,self.pad_x),pady=(self.button_pad,0), stick="news")
        rowcnt = rowcnt+1

        self.cluster_buttton = ctk.CTkButton(master = self.control_frame,
                               text="Compute Clusters",
                               font=self.font,
                               height=button_height,
                               command=self.compute_clusters)
        self.cluster_buttton.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.axis_name_title = ctk.CTkLabel(self.control_frame, text="Choose Axis Names:",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.axis_name_title.grid(row=rowcnt, column=0, columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.axis_name_frame = ScrollableInputFrame(master=self.control_frame,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    height=150,
                                                    )
        self.axis_name_frame._scrollbar.configure(height=50)
        self.axis_name_frame.grid(row=rowcnt, column=0, columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="ew")
        rowcnt = rowcnt+1

        label = ctk.CTkLabel(self.control_frame, text="Choose algorithm",
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1
        
        self.algo_switch = ctk.CTkSwitch(master=self.control_frame,
                                         text="",
                                         command=self.algo_switch_event,
                                         variable=self.session.settings_vars["algorithm"],
                                         onvalue="Hierarchy", offvalue="Witness")
        self.algo_switch.grid(row=rowcnt, column=1, padx=(self.pad_x_inter,self.pad_x), pady=(self.button_pad,0), sticky="news")
        label = ctk.CTkLabel(self.control_frame, textvariable=self.session.settings_vars["algorithm"],
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0, padx=(self.pad_x,0), pady=(self.pad_y_inter,0), sticky="news")
        rowcnt = rowcnt+1

        self.slider_label = ctk.CTkLabel(self.control_frame, text="Choose relative distance eps",
                             justify='center',
                             font=self.font,
                             anchor="center")
        rowcnt = rowcnt+1

        self.slider = ctk.CTkSlider(master=self.control_frame, from_=0, to=1, number_of_steps=50, variable=self.session.settings_vars["eps"])
        self.slider_display = ctk.CTkEntry(master=self.control_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["eps"],
                                    justify='center',
                            )
        self.eps_cnt = rowcnt
        rowcnt = rowcnt+1
        self.algo_switch_event()  # positioning is done in here

        self.plot_title = ctk.CTkLabel(self.control_frame, text="Plot Setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.pad_y_inter,0), sticky="news")
        rowcnt = rowcnt+1

        frame = ctk.CTkFrame(master=self.control_frame)
        frame.grid(column=0,row=rowcnt, columnspan=2, sticky="snwe",pady=(self.button_pad,0), padx=(self.pad_x, self.pad_x))
        frame.grid_columnconfigure(0,weight=1,uniform=1)
        frame.grid_columnconfigure(1,weight=1,uniform=1)
        button = ctk.CTkButton(master = frame,
                               text="Add plot",
                               height=button_height,
                               font=self.font,
                               command=self.add_plot)
        button.grid(row=0,column=0,padx=(0,2.5),pady=(0,0), sticky="news")
        button = ctk.CTkButton(master = frame,
                               text="Remove plot",
                               height=button_height,
                               font=self.font,
                               command=self.remove_plot)
        button.grid(row=0,column=1,padx=(2.5,0),pady=(0,0), sticky="news")
        rowcnt = rowcnt+1
        self.plot_selection = ScrollablePlotSelectFrame(master=self.control_frame,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    height=100,
                                                    )
        self.plot_selection._scrollbar.configure(height=50)
        self.plot_selection.grid(row=rowcnt, column=0, columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="ew")
        rowcnt = rowcnt+1

        self.plot_point_slider_label = ctk.CTkLabel(self.control_frame, text="Number of points to be plotted",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.plot_point_slider_label.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.plot_point_slider = ctk.CTkSlider(master=self.control_frame,
                                               from_=0,
                                               to=1,
                                               number_of_steps=50,
                                               variable=self.session.settings_vars["num_plot_points"])
        self.plot_point_slider.grid(row=rowcnt, column=0, padx=(self.pad_x,0), pady=(self.button_pad,0), sticky="news")
        self.plot_point_slider_display = ctk.CTkEntry(master=self.control_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["num_plot_points"],
                                    justify='center',
                            )
        self.plot_point_slider_display.grid(row=rowcnt, column=1, padx=(self.pad_x_inter,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.compute_button = ctk.CTkButton(master = self.control_frame,
                               text="Compute and Plot",
                               height=button_height,
                               font=self.font,
                               command=self.compute)
        self.compute_button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        title = ctk.CTkLabel(self.control_frame, text="Export Setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.chamber_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Chamber details for export",
                               font=self.font,
                               height=button_height,
                               command=self.select_chamber)
        self.chamber_button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        button = ctk.CTkButton(master = self.control_frame,
                               text="Export directory",
                               height=button_height,
                               font=self.font,
                               command=self.choose_output_dir)
        button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        label = ctk.CTkLabel(self.control_frame, textvariable=self.session.settings_vars["output_path"],
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.pad_y_inter,0), sticky="news")
        rowcnt = rowcnt+1

        self.compute_button = ctk.CTkButton(master = self.control_frame,
                               text="Export Results and Plots",
                               height=button_height,
                               font=self.font,
                               command=self.export)
        self.compute_button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1
        
        # configure results
        self.plot_title = ctk.CTkLabel(self.res_frame, text="Results",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=0, column=0, padx=(self.pad_x,self.pad_x), pady=(self.pad_y_inter,self.pad_y_inter), sticky="news")
        
        #############################
        # End Experiments
        #############################

        #############################
        # Settings
        #############################
        self.settings_frame = ctk.CTkScrollableFrame(master=self.tab_settings, width=400)
        self.settings_frame.grid(column=0,row=0, sticky="snwe",pady=(5,0), padx=(5,5))
        self.settings_frame.grid_columnconfigure(0,weight=1,uniform=1)

        rowcnt=0
        tmp_title = ctk.CTkLabel(self.settings_frame, text="Settings",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="ew")
        rowcnt = rowcnt+1


        tmp_textbox = ctk.CTkTextbox(self.settings_frame,
                             font=self.font,
                             wrap='word',
                             height=150,
                             )
        tmp_textbox.grid(row=rowcnt, column=0, columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="news")
        tmp_textbox.tag_config("center", justify="center")
        tmp_textbox.insert(0.0,text=r"""
                            All settings are stored in the folder $HOME/.pcr_conc, hence to copy 
                            settings from one machine to another, just copy that folder to the desired machine.

                            To restore default settings, just delete the folder $HOME/.pcr_conc. 
                            But note that this also resets the axis names to the default names. 
                            If you want to reset only the settings, delete the file $HOME/.pcr_conc/settings.csv.
                            """
                           )
        rowcnt = rowcnt+1

        tmp_title = ctk.CTkLabel(self.settings_frame, text="General computation",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="ew")
        rowcnt = rowcnt+1

        tmp_textbox = ctk.CTkTextbox(self.settings_frame,
                             font=self.font,
                             wrap='word',
                             )
        tmp_textbox.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="news")
        tmp_textbox.tag_config("center", justify="center")
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
        self.outliers.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        self.outliers_slider = ctk.CTkSlider(master=self.settings_frame, from_=0, to=0.1, number_of_steps=5, variable=self.session.settings_vars["outliers"])
        self.outliers_slider.grid(row=rowcnt, column=0, padx=(self.pad_x,0), pady=(self.button_pad,0), sticky="news")
        self.outliers_slider_display = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["outliers"],
                                    justify='center',
                            )
        self.outliers_slider_display.grid(row=rowcnt, column=1, padx=(self.pad_x_inter,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_title = ctk.CTkLabel(self.settings_frame, text="Negative control detection",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        tmp_textbox = ctk.CTkTextbox(self.settings_frame,
                             font=self.font,
                             wrap='word',
                             )
        tmp_textbox.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="news")
        tmp_textbox.insert(0.0,text="""
                            We ignore the `maximal possible positive points` many 
                            """
                           )
        rowcnt = rowcnt+1

        label = ctk.CTkLabel(self.settings_frame, text="Threshold for negative dimensions (if on one dimension all points execept 'max_positives' are below, dimension is considered negative).",
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        slider = ctk.CTkSlider(master=self.settings_frame, from_=1, to=50000, number_of_steps=500, variable=self.session.settings_vars["neg_threshold"])
        slider.grid(row=rowcnt, column=0, padx=(self.pad_x,0), pady=(self.button_pad,0), sticky="news")
        slider = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["neg_threshold"],
                                    justify='center',
                            )
        slider.grid(row=rowcnt, column=1, padx=(self.pad_x_inter,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        label = ctk.CTkLabel(self.settings_frame, text="Maximal possible positive points for a dimension to be condidered negative.",
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        slider = ctk.CTkSlider(master=self.settings_frame, from_=1, to=10, number_of_steps=11, variable=self.session.settings_vars["negatives_max_positives"])
        slider.grid(row=rowcnt, column=0, padx=(self.pad_x,0), pady=(self.button_pad,0), sticky="news")
        slider = ctk.CTkEntry(master=self.settings_frame,
                                font=self.font,
                                textvariable=self.session.settings_vars["negatives_max_positives"],
                                    justify='center',
                            )
        slider.grid(row=rowcnt, column=1, padx=(self.pad_x_inter,self.pad_x), pady=(self.button_pad,0), sticky="news")
        rowcnt = rowcnt+1

        #tmp_title = ctk.CTkLabel(self.settings_frame, text="Export Settings",
        #                     justify='center',
        #                     font=self.titlefont,
        #                     anchor="center")
        #tmp_title.grid(row=rowcnt, column=0,columnspan=2, padx=(self.pad_x,self.pad_x), pady=(5,0), sticky="news")
        #rowcnt = rowcnt+1

        #button = ctk.CTkButton(master = self.settings_frame,
                               #text="Save as default",
                               #height=button_height,
                               #font=self.font,
                               #command=self.save_default)
        #button.grid(row=rowcnt,column=0, columnspan=2,padx=(self.pad_x,self.pad_x),pady=(self.button_pad + 20,0), sticky="news")
        #rowcnt = rowcnt+1
        
        
        #########################
        # END SETTINGS
        ########################

        self.root.protocol("WM_DELETE_WINDOW", self.destroy_root)

        self.root.mainloop()
    
    def add_plot(self):
        self.plot_selection.add_item()
    
    def remove_plot(self):
        self.plot_selection.remove_last()
   
    def algo_switch_event(self):
        
        # force recompute
        self.session.decision = None
        self.cluster_buttton.configure(require_redraw=True, fg_color=self.orig_button_color)

        if self.session.settings_vars["algorithm"].get() == "Hierarchy":
            self.slider.grid_forget()
            self.slider_label.grid_forget()
            self.slider_display.grid_forget()
            self.slider.configure(state="disabled")
            self.slider_display.configure(state="disabled")
        else:
            self.slider.configure(state="normal")
            self.slider_display.configure(state="normal")
            self.slider.grid(row=self.eps_cnt, column=0,
                             padx=(self.pad_x,0),
                             pady=(self.button_pad,0),
                             sticky="news")
            self.slider_display.grid(row=self.eps_cnt,
                                     column=1, 
                                     padx=(self.pad_x_inter,self.pad_x),
                                     pady=(self.button_pad,0), sticky="news")
            self.slider_label.grid(row=self.eps_cnt-1,
                                   column=0,
                                   columnspan=2,
                                   padx=(self.pad_x,self.pad_x),
                                   pady=(self.button_pad,0), sticky="news")
   
   
    
    def destroy_root(self):
        self.root.quit()
        
    def select_chamber(self):
        try:
            # take parents to match expected file structure
            temp_path = Path(self.session.settings_vars["input_path"].get()).parent.parent.absolute()
            paths = fd.askopenfile(parent=self.root, filetypes=[("csv files", "*.csv")], title="Choose chamber details file", initialdir=str(temp_path))
            self.session.get_chamber_file(paths)
            self.chamber_button.configure(require_redraw=True, fg_color="green")
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def select_files(self):
        try:
            temp_path = Path(self.session.settings_vars["input_path"].get()).parent.parent.parent.absolute()
            paths = fd.askopenfilenames(parent=self.root, filetypes=[("csv files", "*.csv")], title="Choose files to be processed", initialdir=str(temp_path))
            self.session.get_files(paths, axis_frame=self.axis_name_frame, select_frame=self.plot_selection)
            self.files_button.configure(require_redraw=True, fg_color="green")
            if self.session.neg_data is None:
                self.neg_button.configure(require_redraw=True, fg_color=self.orig_button_color)
            if self.session.decision is None:
                self.cluster_buttton.configure(require_redraw=True, fg_color=self.orig_button_color)
            if self.session.chamber_file is None:
                self.chamber_button.configure(require_redraw=True, fg_color=self.orig_button_color)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def choose_output_dir(self):
        try:
            out_dir = fd.askdirectory(parent=self.root, title="Output directory", initialdir=self.session.settings_vars["output_path"].get())
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
            temp_path = Path(self.session.settings_vars["input_path"].get()).absolute()
            paths = fd.askopenfilenames(parent=self.root, filetypes=[("csv files", "*.csv")], title="Select Negative Controls", initialdir=str(temp_path))
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
        try:
            fig, df_results = self.session.compute(self.axis_name_frame, self.plot_selection)
            self.cluster_buttton.configure(require_redraw=True, fg_color="green")
            self.plot_point_slider._to = self.session.decision.X_transformed.shape[0]
            self.session.store_settings(axis=True, settings=False, key="eps")
            self.session.store_settings(axis=True, settings=False, key="algorithm")
            self.draw_results(df_results)
            self.draw_figure(fig)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def export(self):
        try:
            self.session.export(self.axis_name_frame, self.plot_selection)
            self.session.store_settings()
            msg(title="Export", message="Successful export.", icon="check")
        except Exception as e:
            self.session.chamber_file = None
            self.chamber_button.configure(require_redraw=True, fg_color=self.orig_button_color)
            msg(title="Error", message=str(e), icon="cancel")
        
    def save_default(self):
        try:
            self.session.store_settings()
            msg(title="Export", message="Successfully saved settings as default.", icon="check")
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")
        
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