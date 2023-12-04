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
import pandas as pd
import time

class ctkApp:
        
    def __init__(self):
        

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.geometry("1200x800+200x200")
        self.root.title("PCR Contamination")
        self.root.update()

        self.session = TkinterSession()
        self.session.eps = tk.DoubleVar(self.root, value=0.4, name="eps")

        self.font = (tk.font.nametofont("TkDefaultFont"), 15)
        self.titlefont = (tk.font.nametofont("TkDefaultFont"), 20)
        bg_color = self.root.cget("fg_color")

        self.root.grid_columnconfigure(1,weight=1,uniform=1)
        self.root.grid_rowconfigure(0,weight=10)
        self.root.grid_rowconfigure(1,weight=2)

        self.plot_frame = ctk.CTkFrame(master=self.root)
        self.plot_frame.grid(column=1,row=0, sticky="snwe",pady=(5,5), padx=(5,5))
        self.plot_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.plot_frame.grid_rowconfigure(0,weight=1,uniform=1)

        self.res_frame = ctk.CTkFrame(master=self.root)
        self.res_frame.grid(column=1,row=1, sticky="we",pady=(5,5), padx=(5,5))
        self.res_frame.grid_columnconfigure(0,weight=1,uniform=1)
        self.res_frame.grid_rowconfigure(1,weight=1)
        self.res_frame.grid_rowconfigure(0,weight=1)

        self.control_frame = ctk.CTkFrame(master=self.root,
                                  width = 400,
                                  fg_color=bg_color,
                                  )
        self.control_frame.grid(column=0,row=0,rowspan=2, sticky="sn",pady=(5,5), padx=(5,5))
        self.control_frame.grid_columnconfigure(0,weight=25)
        self.control_frame.grid_columnconfigure(1,weight=15)

        button_pad = 20
        rowcnt = 0

        self.experiment_title = ctk.CTkLabel(self.control_frame, text="Experiment Setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.experiment_title.grid(row=rowcnt, column=0,columnspan=2, padx=(50,50), pady=(50,0), sticky="ew")
        rowcnt = rowcnt+1

        self.files_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Experiment files",
                               width=300,
                               height=50,
                               font=self.font,
                               command=self.select_files)
        self.files_button.grid(row=rowcnt,column=0, columnspan=2,padx=(50,50),pady=(button_pad,0), sticky="ew")
        self.orig_button_color = self.files_button.cget("fg_color")
        rowcnt = rowcnt+1

        self.neg_button = ctk.CTkButton(master = self.control_frame,
                               text="Select Negative control",
                               width=200,
                               height=50,
                               font=self.font,
                               command=self.select_neg)
        self.neg_button.grid(row=rowcnt,column=0,padx=(50,0),pady=(button_pad,0), sticky="we" )

        self.neg_button_cancel = ctk.CTkButton(master = self.control_frame,
                               text="Unselect",
                               width=90,
                               height=50,
                               font=self.font,
                               command=self.cancel_neg)
        self.neg_button_cancel.grid(row=rowcnt,column=1,padx=(10,50),pady=(button_pad,0), stick="ew")
        rowcnt = rowcnt+1

        self.axis_name_title = ctk.CTkLabel(self.control_frame, text="Choose Axis Names:",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.axis_name_title.grid(row=rowcnt, column=0, columnspan=2, padx=(50,50), pady=(button_pad,0), sticky="we")
        rowcnt = rowcnt+1

        self.axis_name_frame = ScrollableInputFrame(master=self.control_frame,
                                                    width=300,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    )
        self.axis_name_frame.grid(row=rowcnt, column=0, columnspan=2, padx=(50,50), pady=(button_pad,0), sticky="we")
        rowcnt = rowcnt+1

        self.slider_label = ctk.CTkLabel(self.control_frame, text="Choose relative distance eps",
                             justify='center',
                             font=self.font,
                             anchor="center")
        self.slider_label.grid(row=rowcnt, column=0,columnspan=2, padx=(50,50), pady=(button_pad,0), sticky="we")
        rowcnt = rowcnt+1

        self.slider = ctk.CTkSlider(master=self.control_frame, from_=0, to=1, width=200, number_of_steps=50, variable=self.session.eps)
        self.slider.grid(row=rowcnt, column=0, padx=(50,0), pady=(button_pad,0), sticky="we")
        self.slider_display = ctk.CTkEntry(master=self.control_frame,
                                font=self.font,
                                textvariable=self.session.eps,
                                    justify='center',
                                width=90,
                            )
        self.slider_display.grid(row=rowcnt, column=1, padx=(10,50), pady=(button_pad,0), sticky="we")
        rowcnt = rowcnt+1

        self.plot_title = ctk.CTkLabel(self.control_frame, text="Plot setup",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=rowcnt, column=0,columnspan=2, padx=(50,50), pady=(5,0), sticky="we")
        rowcnt = rowcnt+1

        self.plot_selection = ScrollablePlotSelectFrame(master=self.control_frame,
                                                    width=300,
                                                    corner_radius=0,
                                                    font=self.font,
                                                    )
        self.plot_selection.grid(row=rowcnt, column=0, columnspan=2, padx=(50,50), pady=(button_pad,0), sticky="ew")
        rowcnt = rowcnt+1

        self.compute_button = ctk.CTkButton(master = self.control_frame,
                               text="Compute",
                               width=300,
                               height=50,
                               font=self.font,
                               command=self.compute)
        self.compute_button.grid(row=rowcnt,column=0, columnspan=2,padx=(50,50),pady=(button_pad,0), sticky="we")
        rowcnt = rowcnt+1
        
        # configure results
        self.plot_title = ctk.CTkLabel(self.res_frame, text="Results",
                             justify='center',
                             font=self.titlefont,
                             anchor="center")
        self.plot_title.grid(row=0, column=0, padx=(50,50), pady=(5,5), sticky="nswe")

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
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def select_neg(self):
        try:
            paths = fd.askopenfilenames(parent=self.root, filetypes=[("csv files", "*.csv")], title="Select Negative Controls", initialdir="/home/nico/csem/lab/Data/6P-positive-dilution-series-1-unlabelled/droplet-level-data/RawData")
            self.session.get_negs(paths)
            self.neg_button.configure(require_redraw=True, fg_color="green")
            msg(title="Valid", message="Valid negative control.", icon="check")
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def cancel_neg(self):
        try:
            self.session.drop_negs()
            self.neg_button.configure(require_redraw=True, fg_color=self.orig_button_color)
        except Exception as e:
            msg(title="Error", message=str(e), icon="cancel")

    def compute(self):
        try:
            fig, df_results = self.session.compute(self.axis_name_frame, self.plot_selection)
            self.draw_results(df_results)
            self.draw_figure(fig)
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