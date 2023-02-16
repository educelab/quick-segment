#Volume Warp Page within the GUI
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QMessageBox, QMenuBar, QMenu
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

import qs.apps.quick_segment

from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_interpolation)
from qs.math import find_min

# -------------------------------------------------------------------
#                             SEGMENT WINDOW CLASS
# -------------------------------------------------------------------
class WarpWindow(QtWidgets.QWidget):
    ax = None
    bar = None

    def __init__(self, vol, seg_dir, initial_slice):
        super().__init__()

        # -------------initial window specs---------------
        # StackedWindow.window_set('Volume Warping')
        self.vol = vol