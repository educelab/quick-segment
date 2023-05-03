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
        self.test_label = QtWidgets.QLabel()
        click_vid = QtGui.QMovie('qs/test_click.gif')
        #around 58:45 x20
        click_vid.setScaledSize(QtCore.QSize(580,450))
        self.test_label.setMovie(click_vid)
        click_vid.start()
        
        self.description_layout = QtWidgets.QVBoxLayout()
        self.description_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        #----add in discription----
        self.topic_label = QtWidgets.QLabel("\nAdding Points")
        big_font = QtGui.QFont("San Francisco", 20)
        big_font.setBold(True)
        self.topic_label.setFont(big_font)
        # self.topic_label.setMaximumHeight(20)
        self.descript_label = QtWidgets.QLabel("You can add points to a slice by clicking anywhere on the canvas")
        # self.descript_label.setMaximumHeight(20)
        self.description_layout.addWidget(self.topic_label)
        self.description_layout.addWidget(self.descript_label)
        
        window_layout.addWidget(self.test_label)
        window_layout.addLayout(self.description_layout)
        
        
        
        