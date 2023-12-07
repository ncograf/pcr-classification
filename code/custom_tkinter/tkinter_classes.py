import customtkinter as ctk
import tkinter
from icecream import ic
from typing import List, Dict

class ScrollableInputFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, font, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.radiobutton_variable = ctk.StringVar()
        self.input_list = []
        self.label_list = []
        self.font = font

    def add_item(self, label : str, default : Dict[str, str]):
        if not default is None and label in default.keys():
            # using textvars will retain the value entered for each axis even if the data is newly loaded
            textvar = tkinter.StringVar(self, name=label, value=default[label])
        else:
            textvar = tkinter.StringVar(self, name=label)
        label = ctk.CTkLabel(self, text=label,
                             justify='center',
                             font=self.font,
                             anchor="center")
        label.grid(row=len(self.input_list), column=0, padx=(20,0), pady=(5, 5), sticky="w")
        input =  ctk.CTkEntry(master=self,
                             font=self.font,
                             textvariable=textvar,
                                justify='center',
                            )
        
        input.grid(row=len(self.input_list), column=1, pady=(5, 5), sticky="w")
        self.input_list.append(input)
        self.label_list.append(label)

    def remove_all(self):
        for i in self.label_list:
            i.destroy()
           
        for j in self.input_list:
            j.destroy()
        
        self.label_list = []
        self.input_list = []

class ScrollablePlotSelectFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, font, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.radiobutton_variable = ctk.StringVar()
        self.labels = []
        self.left_list = []
        self.right_list = []
        self.font = font
    
    def set_labels(self, labels : List[str]):
        self.labels = labels

    def add_item(self, init_left : int = None, init_right : int = None):
        plot_num = len(self.left_list)
        if init_left is None:
            text_left = tkinter.StringVar(self, name=f"{plot_num}_left")
        else:
            text_left = tkinter.StringVar(self, self.labels[init_left], name=f"{plot_num}_left")

        if init_right is None:
            text_right = tkinter.StringVar(self, name=f"{plot_num}_right")
        else:
            text_right = tkinter.StringVar(self, self.labels[init_right], name=f"{plot_num}_right")

        left = ctk.CTkComboBox(self,
                             font=self.font,
                             values=self.labels,
                             variable=text_left,
        )
        left.grid(row=len(self.left_list), column=0, padx=(20,0), pady=(5, 5), sticky="w")

        right = ctk.CTkComboBox(self,
                             font=self.font,
                             values=self.labels,
                             variable=text_right,
        )
        right.grid(row=len(self.right_list), column=1, padx=(20,0), pady=(5, 5), sticky="w")

        self.right_list.append(right)
        self.left_list.append(left)

    def remove_all(self):
        for i in self.left_list:
            i.destroy()
           
        for j in self.right_list:
            j.destroy()
        
        self.right_list = []
        self.left_list = []