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

# noinspection PyUnresolvedReferences
import qs.resources

import qs.apps.segmentation as SegPage
import qs.apps.volume_warp as WarpPage

from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_interpolation)
from qs.math import find_min



# -------------------------------------------------------------------
#                             STACKED WINDOW CLASS
# -------------------------------------------------------------------
class StackedWindow(QtWidgets.QWidget):
    ax = None
    bar = None

    def __init__(self, vol, seg_dir, initial_slice=0):
        super().__init__()

        #------------intial window set up-----------
        self.window_width = 1100
        self.window_height = 900
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Interpolation Segmentation")

        #--------------------page navigation----------------
        #implimented using stacked widgets
        #making the stack
        self.Stack = QtWidgets.QStackedWidget(self)
        #making stack widgets from each window class
        self.segment_page = SegPage.SegmentWindow(vol, seg_dir, initial_slice)
        self.warp_page = WarpPage.WarpWindow(vol, seg_dir, initial_slice)
        #adding widgets to the stack
        self.Stack.addWidget(self.segment_page)
        self.Stack.addWidget(self.warp_page)

        self.Stack.setCurrentWidget(self.segment_page)

        #--------------------------menu bar---------------------
        self.menubar = QMenuBar()
        self.pages = self.menubar.addMenu('Pages')
        self.pages.addAction('Segmentation', lambda: self.Stack.setCurrentWidget(self.segment_page))
        self.pages.addAction('Warping', lambda: self.Stack.setCurrentWidget(self.warp_page))

    #sets the window specs
    def window_set(self, title):
        self.setWindowTitle("Interpolation Segmentation")



# -----------------------------------------------------------------
#                           MAIN
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--input-volpkg", required=True)
    parser.add_argument("--volume", type=int, default=0)
    args = parser.parse_args()

    # ---------saving paths to folders within volume------------
    volpkg_path = Path(args.input_volpkg)
    volumes_dir = volpkg_path / 'volumes'
    segmentation_dir = volpkg_path / 'paths'
    volumes = sorted([x for x in volumes_dir.iterdir() if x.is_dir()])
    input_vol_dir = volumes[args.volume]

    # ----------------loading Zarr OR Volume------------------
    # Zarr = new volume representation -> Only loads chuncks which are needed = saves memory and is faster
    # Code from Stephen's volume.py (ink-id)
    start = time.time()
    vol = Volume.from_path(str(input_vol_dir))
    end = time.time()
    print(f"{end - start} seconds to initialize {vol.shape} volume")

    # creating and loading appliaction window
    app = QtWidgets.QApplication(sys.argv)
    window = StackedWindow(vol, segmentation_dir)
    window.show()
    # allows for exit from the application
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')


if __name__ == "__main__":
    main()
