from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimediaWidgets, QtMultimedia
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer

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
        self.window_width = 550
        self.window_height = 750
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Quick Segment Tutorial")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # ------------------------------Window GUI-----------------------------
        # Overall Window layout
        window_layout = QtWidgets.QHBoxLayout()

        #create tabs widget 
        self.tabs = QtWidgets.QTabWidget()
        
        #----Generation of Info Tabs----     
        #----Set big font---   
        big_font = QtGui.QFont("San Francisco", 20) #does not like San Fanciso font
        big_font.setBold(True)
        
        #-----------------------Loading in mp4s-------------------------v
        #----shadows-----
        self.s_vid_layout = QtWidgets.QVBoxLayout()
        self.s_vid_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        self.s_player = QMediaPlayer()
        self.s_player.setSource(QtCore.QUrl('qs/mp4s/shadows.mp4'))
        self.s_vidWid = QVideoWidget()
        self.s_vidWid.setMaximumSize(550, 600)
        self.s_player.setVideoOutput(self.s_vidWid)
        self.s_vidWid.show()
        self.s_player.play()
        self.s_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.s_vid_layout.addWidget(self.s_vidWid)
        
        #----interpolation-----
        self.i_vid_layout = QtWidgets.QVBoxLayout()
        self.i_vid_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        self.i_player = QMediaPlayer()
        self.i_player.setSource(QtCore.QUrl('qs/mp4s/interpolation.mp4'))
        self.i_vidWid = QVideoWidget()
        self.i_vidWid.setMaximumSize(550, 600)
        self.i_player.setVideoOutput(self.i_vidWid)
        self.i_vidWid.show()
        self.i_player.play()
        self.i_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.i_vid_layout.addWidget(self.i_vidWid)
        
        #----slice nav-----
        self.sn_vid_layout = QtWidgets.QVBoxLayout()
        self.sn_vid_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        self.sn_player = QMediaPlayer()
        self.sn_player.setSource(QtCore.QUrl('qs/mp4s/slice_nav.mp4'))
        self.sn_vidWid = QVideoWidget()
        self.sn_vidWid.setMaximumSize(550, 600)
        self.sn_player.setVideoOutput(self.sn_vidWid)
        self.sn_vidWid.show()
        self.sn_player.play()
        self.sn_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.sn_vid_layout.addWidget(self.sn_vidWid)
        
        #----segmentation loading-----
        self.sl_vid_layout = QtWidgets.QVBoxLayout()
        self.sl_vid_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        self.sl_player = QMediaPlayer()
        self.sl_player.setSource(QtCore.QUrl('qs/mp4s/seg_loading.mp4'))
        self.sl_vidWid = QVideoWidget()
        self.sl_vidWid.setMaximumSize(600, 600)
        self.sl_player.setVideoOutput(self.sl_vidWid)
        self.sl_vidWid.show()
        self.sl_player.play()
        self.sl_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.sl_vid_layout.addWidget(self.sl_vidWid)
        #-----------------------Loading in mp4s-------------------------^
        
        #--------------------------------General Click------------------------------
        self.gc_label = QtWidgets.QLabel("\nAdding Points")
        self.gc_label.setFont(big_font)
        self.gc_descript_label = QtWidgets.QLabel(" - Left click on the canvas to add points")
        
        #----generate gif------
        self.gcv_label = QtWidgets.QLabel()
        click_vid = QtGui.QMovie('qs/gifs/click.gif')
        #around 58:45 x20
        # click_vid.setScaledSize(QtCore.QSize(580,450))
        self.gcv_label.setMovie(click_vid)
        click_vid.start()
                                
        #add layout for info section
        self.gc_layout = QtWidgets.QVBoxLayout()
        self.gc_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.gc_layout.addWidget(self.gc_label)
        self.gc_layout.addWidget(self.gc_descript_label)
        
        #create tab widget
        self.page_1 =QtWidgets.QWidget()
        self.p1_layout =  QtWidgets.QVBoxLayout()
        self.p1_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p1_layout.addLayout(self.gc_layout)
        self.p1_layout.addWidget(self.gcv_label) #click vid
        # self.p1_layout.addWidget(self.sn_vidWid) #click vid
        self.page_1.setLayout(self.p1_layout)
        
        #add tab to list of tabs
        self.p1_index = self.tabs.addTab(self.page_1, "Points")
        
        #--------------------------------Long Click Move-------------------------------->move
        self.lcm_label = QtWidgets.QLabel("\nMoving Points")
        self.lcm_label.setFont(big_font)
        self.lcm_descript_label = QtWidgets.QLabel(" - Left click on a point then drag and drop it where you want to move it to \n\nNote: A point selected to move turns blue")
        
        #----generate gif------
        self.lmcv_label = QtWidgets.QLabel()
        move_vid = QtGui.QMovie('qs/gifs/move.gif')
        #around 58:45 x20
        move_vid.setScaledSize(QtCore.QSize(600,470))
        self.lmcv_label.setMovie(move_vid)
        move_vid.start()
        
        #add layout for info section
        self.lmc_layout = QtWidgets.QVBoxLayout()
        self.lmc_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.lmc_layout.addWidget(self.lcm_label)
        self.lmc_layout.addWidget(self.lcm_descript_label)
        
        #create tab widget
        self.page_2 =QtWidgets.QWidget()
        self.p2_layout =  QtWidgets.QVBoxLayout()
        self.p2_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p2_layout.addLayout(self.lmc_layout)
        self.p2_layout.addWidget(self.lmcv_label) #move vid
        # self.p2_layout.addWidget(self.sn_vidWid) #move vid
        self.page_2.setLayout(self.p2_layout)
        
        #add tab to list of tabs
        self.p2_index = self.tabs.addTab(self.page_2, "Adjust")
        #---------------------------------Canvas Zoom/pan---------------------------------
        self.c_label = QtWidgets.QLabel("\nCanvas Navigation")
        self.c_label.setFont(big_font)
        self.c_descript_label = QtWidgets.QLabel(" - Scrolling zooms in and out of the canvas\n - Left click hold drag and release allows you to pan")
        
        #----generate gif------
        self.cv_label = QtWidgets.QLabel()
        cv_vid = QtGui.QMovie('qs/gifs/zoom_pan.gif')
        #around 58:45 x20
        cv_vid.setScaledSize(QtCore.QSize(600,470))
        self.cv_label.setMovie(cv_vid)
        cv_vid.start()
        
        #add layout for info section
        self.c_layout = QtWidgets.QVBoxLayout()
        self.c_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.c_layout.addWidget(self.c_label)
        self.c_layout.addWidget(self.c_descript_label)
        
        #create tab widget
        self.page_3 =QtWidgets.QWidget()
        self.p3_layout =  QtWidgets.QVBoxLayout()
        self.p3_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p3_layout.addLayout(self.c_layout)
        self.p3_layout.addWidget(self.cv_label) #zoom/pan vid
        self.page_3.setLayout(self.p3_layout)
        
        #add tab to list of tabs
        self.p3_index = self.tabs.addTab(self.page_3, "Canvas Nav")
        
        #---------------------------------Shadows---------------------------------
        self.s_label = QtWidgets.QLabel("\nShadows")
        self.s_label.setFont(big_font)
        self.s_descript_label = QtWidgets.QLabel(" - A shadow is your points from another key slice\n     - Black shadows are for prior key slices \n     - White shadows are for the succeeding key slices\n - Circles surrounding the points on the shadow help you keep track of how many points you have placed")
        
        #add layout for info section
        self.s_layout = QtWidgets.QVBoxLayout()
        self.s_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.s_layout.addWidget(self.s_label)
        self.s_layout.addWidget(self.s_descript_label)
        
        #create tab widget
        self.page_4 =QtWidgets.QWidget()
        self.p4_layout =  QtWidgets.QVBoxLayout()
        self.p4_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p4_layout.addLayout(self.s_layout)
        self.p4_layout.addLayout(self.s_vid_layout) #shadows vid
        self.page_4.setLayout(self.p4_layout)
        
        #add tab to list of tabs
        self.p4_index = self.tabs.addTab(self.page_4, "Shadows")
        
        #---------------------------------Interpolation---------------------------------
        self.i_label = QtWidgets.QLabel("\nInterpolation")
        self.i_label.setFont(big_font)
        self.i_descript_label = QtWidgets.QLabel(" - Between 2 red key slices there are interpolated segmentation lines represented by the yellow color")
                
        #add layout for info section
        self.i_layout = QtWidgets.QVBoxLayout()
        self.i_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.i_layout.addWidget(self.i_label)
        self.i_layout.addWidget(self.i_descript_label)
        
        #create tab widget
        self.page_5 =QtWidgets.QWidget()
        self.p5_layout =  QtWidgets.QVBoxLayout()
        self.p5_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p5_layout.addLayout(self.i_layout)
        self.p5_layout.addLayout(self.i_vid_layout) #interp vid
        self.page_5.setLayout(self.p5_layout)
        
        #add tab to list of tabs
        self.p5_index = self.tabs.addTab(self.page_5, "Interpolation")
        
        #---------------------------------Slice Navigation---------------------------------
        #need to split this into different pages for shorter gifs to lead
        
        self.sn_label = QtWidgets.QLabel("\nSlice Navigation")
        self.sn_label.setFont(big_font)
        self.sn_descript_label = QtWidgets.QLabel(" - Single arrows move a single slice forward and backwards\n - Double arrows move to the nearest key slice in the given direciton,\n   when no key slice the number of slices indicated by the jump size box \n - The dropdown allows for viewing all of the key slices and navigate to them")
        
        #add layout for info section
        self.sn_layout = QtWidgets.QVBoxLayout()
        self.sn_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.sn_layout.addWidget(self.sn_label)
        self.sn_layout.addWidget(self.sn_descript_label)
        
        #create tab widget
        self.page_6 = QtWidgets.QWidget()
        self.p6_layout =  QtWidgets.QVBoxLayout()
        self.p6_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p6_layout.addLayout(self.sn_layout)
        self.p6_layout.addLayout(self.sn_vid_layout) #slice nav vid
        self.page_6.setLayout(self.p6_layout)
        
        #add tab to list of tabs
        self.p6_index = self.tabs.addTab(self.page_6, "Slice Nav")
        
        #---------------------------------Segmentation Loading---------------------------------
        self.sl_label = QtWidgets.QLabel("\nSegmentation Loading and Editing")
        self.sl_label.setFont(big_font)
        self.sl_descript_label = QtWidgets.QLabel(" - Previous segmentations can be viewed and edited by selecting on them in the previous \n segmentations box \n - Right click on segmentation select it for editing, indicated by extra circle \n - Add your current segmentation to the list by saving points \n\nNote: It is strongly recommended to load segmentations completed through the Quick Segment tool, \nsegmentations from other tools may not be compatible ")
        
        #add layout for info section
        self.sl_layout = QtWidgets.QVBoxLayout()
        self.sl_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.sl_layout.addWidget(self.sl_label)
        self.sl_layout.addWidget(self.sl_descript_label)
        
        #create tab widget
        self.page_7 =QtWidgets.QWidget()
        self.p7_layout =  QtWidgets.QVBoxLayout()
        self.p7_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.p7_layout.addLayout(self.sl_layout)
        self.p7_layout.addLayout(self.sl_vid_layout) #interp vid
        self.page_7.setLayout(self.p7_layout)
        
        #add tab to list of tabs
        self.p7_index = self.tabs.addTab(self.page_7, "Seg Load")
        
        
        #----button to easly move to next page?------- -> not in use yet
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(lambda: self.next_tut_page())
        
        #adding tabs and windows to main window
        window_layout.addWidget(self.tabs)
        self.setLayout(window_layout)

        
    def next_tut_page(self):
        self.tabs.setCurrentIndex(self.p2_index)
        
        
        
        