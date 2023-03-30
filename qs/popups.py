from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import plotly.graph_objects as go
import cv2 as cv
import numpy as np
from qs.math import find_sobel_edge
from qs.interpolation import full_interpolation, verify_full_interpolation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

class MyPopup(QtWidgets.QWidget):
    def __init__(self, image: np.array):
        super().__init__()
        self.window_width = 400
        self.window_height = 400
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Advanced Settings")
        self.canvas = FigCanvas(
            plt.Figure(figsize=(5, 4), facecolor='#3d3d3d'))
        self.ax = self.canvas.figure.subplots()
        self.ax.tick_params(labelcolor='white', colors='white')

        # set images
        self.source_img = image.astype('float64')
        self.source_img *= (255.0/self.source_img.max())
        self.source_img = np.uint8(self.source_img)
        self.edge_treated_img = cv.Canny(image=self.source_img, threshold1=100, threshold2=110)
        self.ax.imshow(self.edge_treated_img)

        # Set Layout
        popup_layout = QtWidgets.QHBoxLayout()
        self.setLayout(popup_layout)

        # Explainer box
        explainer_box = QtWidgets.QGroupBox()
        explainer_layout = QtWidgets.QVBoxLayout()
        explainer_box.setLayout(explainer_layout)
        explainer_layout.addWidget(QtWidgets.QLabel(""))

        # Tools box
        tools_box = QtWidgets.QGroupBox()
        tools_layout = QtWidgets.QVBoxLayout()
        tools_box.setLayout(tools_layout)
        # Edge limit text box
        self.edge_limit = QtWidgets.QLineEdit()
        self.edge_limit.setMaxLength(5)
        self.edge_limit.setPlaceholderText("0")
        self.edge_limit.returnPressed.connect(
            lambda: self.set_edge_limit(int(self.edge_limit.text())))
        
        edge_limit_layout = QtWidgets.QHBoxLayout()
        edge_limit_layout.addWidget(QtWidgets.QLabel("Edge value:"))
        edge_limit_layout.addWidget(self.edge_limit)
        tools_layout.addLayout(edge_limit_layout)

        # Edge search regions
        self.edge_search = QtWidgets.QLineEdit()
        self.edge_search.setMaxLength(5)
        self.edge_search.setPlaceholderText("0")
        #self.edge_search.returnPressed.connect(
        #    lambda: self.set_edge_limit(int(self.edge_limit.text())))
        edge_search_layout = QtWidgets.QHBoxLayout()
        edge_search_layout.addWidget(QtWidgets.QLabel("Search size:"))
        edge_search_layout.addWidget(self.edge_search)
        tools_layout.addLayout(edge_search_layout)

        # Set and cancel buttons
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('Save settings')
        self.cancel_button = QtWidgets.QPushButton()
        self.cancel_button.setText('Cancel')
        save_layout = QtWidgets.QHBoxLayout()
        save_layout.addWidget(self.cancel_button)
        save_layout.addWidget(self.save_button)

        # Settings layout
        settings_layout = QtWidgets.QVBoxLayout()
        settings_layout.addWidget(explainer_box)
        settings_layout.addWidget(tools_box)
        settings_layout.addLayout(save_layout)

        popup_layout.addWidget(self.canvas)
        popup_layout.addLayout(settings_layout)

    def set_edge_limit (self, limit):
        """
        Sets the edge based on a limit set by the user

        :param limit: limit of the edge detection
        """

        display_img = np.full_like(self.source_img, 0)
        for row in range(3, self.source_img.shape[0] - 3):
            for pixel in range(3, self.source_img.shape[1] - 3):
                    if (self.edge_treated_img[row][pixel] > limit):
                    #if (find_sobel_edge(self.source_img, [pixel, row]) > limit):
                        display_img[row][pixel] = 10000

        self.ax.imshow(display_img)
        self.canvas.draw_idle()

    def update(self, img):
        self.ax.imshow(img)
        self.canvas.draw_idle()


class ViewPopUp(QtWidgets.QDialog):
    def __init__(self, vol, seg):
        super().__init__()
        self.window_width = 600
        self.window_height = 600
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("View All Points")

        # Store volume info
        self.vol_height = vol.shape[1]
        self.vol_width = vol.shape[2]
        self.vol_slices = vol.shape[0]

        # Set Matplotlib axes
        self.canvas = FigCanvas(
            plt.Figure(figsize=(8, 8), facecolor='#3d3d3d'))
        self.ax = self.canvas.figure.subplots(subplot_kw={'projection':'3d'})
        self.ax.set(facecolor='#3d3d3d')
        self.ax.tick_params(labelcolor='white', colors='white')

        # Slider
        self.slice_slider = QtWidgets.QSlider(Qt.Orientation.Vertical)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(vol.shape[0] - 1)
        self.slice_slider.valueChanged.connect(
            lambda: self.show_slices(vol, self.slice_slider.value(), seg))

        # Show first slice
        self.show_slices(vol, 0, seg)

        #Set Layout
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.canvas)
        layout.addWidget(self.slice_slider)


    def draw_seg(self, seg):
        """
        Draws the segmentation points on the graph

        :param seg: segmentation dictionary
        """
        # if (verify_full_interpolation(seg)):
        #     interpolation = full_interpolation(seg)
        #     for row in interpolation:
        #         for point in row:
        #             self.ax.scatter(point[0],point[1],point[2], c = 'yellow')

        for slice in seg:
            for point in seg[slice]:
                self.ax.scatter(point[0],point[1],point[2], c = 'r', alpha=0.6)

    def show_slices(self, vol, slice, seg):
        """
        Shows the slices of the volume on the 3D graph

        :param vol: volume
        :param slice: current slice on display
        :param seg: segmentation data (only used to draw points)
        """
        self.ax.clear()
        self.ax.set_zlim3d(0, self.vol_slices)
        self.ax.set_ylim3d(-(self.vol_width-self.vol_height)/2, self.vol_height + (self.vol_width-self.vol_height)/2)
        self.ax.set_xlim3d(0, self.vol_width)

        # Find shape values
        x, y = np.meshgrid(np.linspace(0, self.vol_width, self.vol_width), np.linspace(0, self.vol_height, self.vol_height))

        #self.ax.imshow(x, y, vol[0], zdir='z')
        #self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.BrBG(vol[0]), shade=False)
        offset = slice
        self.ax.contourf(x, y, vol[slice],
                         5,
                         zdir='z',
                         offset=offset,
                         alpha=0.9)
        
        self.draw_seg(seg)
        self.canvas.draw()


