from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

class TutorialWindow(QtWidgets.QWidget):
    ax = None
    bar = None

    def __init__(self):
        super().__init__()
        
        # -------------initial window specs---------------
        self.window_width = 900
        self.window_height = 800
        # self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Quick Segment Tutorial")
        

        # ------------------------------Window GUI-----------------------------
        # Overall Window layout
        window_layout = QtWidgets.QHBoxLayout()
        window_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.setLayout(window_layout)
        
        #----add in gif------
        self.video_label = QtWidgets.QLabel()
        click_vid = QtGui.QMovie('qs/basic_tutorial.gif')
        #around 58:45 x20
        # click_vid.setScaledSize(QtCore.QSize(580,450))
        self.video_label.setMovie(click_vid)
        click_vid.start()
        
        #----add in discription----
        self.description_layout = QtWidgets.QVBoxLayout()
        self.description_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        
        big_font = QtGui.QFont("San Francisco", 20)
        big_font.setBold(True)
        #----General Click---->add
        self.gc_label = QtWidgets.QLabel("\nAdding Points")
        self.gc_label.setFont(big_font)
        self.gc_descript_label = QtWidgets.QLabel("Left click on the canvas to add points")
        #----Long Click Move---->move
        self.lcm_label = QtWidgets.QLabel("\nMoving Points")
        self.lcm_label.setFont(big_font)
        self.lcm_descript_label = QtWidgets.QLabel("Left click on a point then drag and drop it where you want to move it to")
        #----Canvas Zoom/pan----
        self.c_label = QtWidgets.QLabel("\nCanvas Navigation")
        self.c_label.setFont(big_font)
        self.c_descript_label = QtWidgets.QLabel(" - Scrolling zooms in and out of the canvas\n - Left click hold drag and release allows you to pan")
        #----Slice Navigation----
        self.sn_label = QtWidgets.QLabel("\nSlice Navigation")
        self.sn_label.setFont(big_font)
        self.sn_descript_label = QtWidgets.QLabel(" - Single arrows move a single slice forward and backwards\n - Double arrows move to the nearest key slice in the given direciton,\n   when no key slice the number of slices indicated by the jump size box \n - The dropdown allows for viewing all of the key slices and navigate to them")
        #----Shadows----
        self.s_label = QtWidgets.QLabel("\nShadows")
        self.s_label.setFont(big_font)
        self.s_descript_label = QtWidgets.QLabel(" - A shadow is your points from another key slice\n     - Black shadows are for prior key slices \n     - White shadows are for the succeeding key slices")
        
        self.description_layout.addWidget(self.gc_label)
        self.description_layout.addWidget(self.gc_descript_label)
        self.description_layout.addWidget(self.lcm_label)
        self.description_layout.addWidget(self.lcm_descript_label)
        self.description_layout.addWidget(self.c_label)
        self.description_layout.addWidget(self.c_descript_label)
        self.description_layout.addWidget(self.sn_label)
        self.description_layout.addWidget(self.sn_descript_label)
        self.description_layout.addWidget(self.s_label)
        self.description_layout.addWidget(self.s_descript_label)
        
        window_layout.addWidget(self.video_label)
        window_layout.addLayout(self.description_layout)
        
        
        
        