from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

# noinspection PyUnresolvedReferences
import qs.resources
from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_partial_interpolation, find_normal_direction, partial_interpolation)
from qs.popups import MyPopup, ViewPopUp
from qs.math import find_min, find_sobel_edge, canny_edge


# -------------------------------------------------------------------
#                             WINDOW CLASS
# ------------------------------------------------------------------
class MainWindow(QtWidgets.QWidget):
    ax = None
    bar = None

    def __init__(self, vol, seg_dir, initial_slice=0):
        super().__init__()

        # -------------initial window specs---------------
        self.window_width = 900
        self.window_height = 800
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Interpolation Segmentation")
        self.colormap = colors.ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.15, 0.85, 256)))
        self.vol = vol

        # ------------------------------Window GUI-----------------------------
        # Overall Window layout
        window_layout = QtWidgets.QHBoxLayout()
        self.setLayout(window_layout)

        # Slice side of GUI layout
        slice_layout = QtWidgets.QVBoxLayout()
        window_layout.addLayout(slice_layout)

        # plot creation and insert
        self.canvas = FigCanvas(
            plt.Figure(figsize=(15, 17), facecolor='#3d3d3d'))

        self.toolbar = NavigationToolbar(self.canvas, self) 
        #removing unnecessary buttons  
        unwanted_buttons = ['Home', 'Save', 'Subplots', 'Customize', 'Zoom']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        #adding widgets to the layout
        slice_layout.addWidget(self.toolbar) #if not added to the layout it is added within the canvas as a collapsed version
        slice_layout.addWidget(self.canvas)
    
        
        self.insert_ax(vol, initial_slice)

        # Toolbar side of GUI layout
        toolbar_layout = QtWidgets.QVBoxLayout()
        window_layout.addLayout(toolbar_layout)

        # ----------------Slice Navigation-------------------
        # Slider label
        self.slider_label = QtWidgets.QLabel("Slice [idx]")
        # Slider
        self.slice_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(initial_slice)
        self.slice_slider.setMaximum(vol.shape[0] - 1)
        self.slice_slider.valueChanged.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        # Slider index box
        self.slice_index = QtWidgets.QLineEdit()
        self.slice_index.setMaxLength(5)
        self.slice_index.setPlaceholderText("0")
        self.slice_index.returnPressed.connect(
            lambda: self.set_slice(vol, self.slice_index.text()))
        # Step slice navigation
        self.big_step_decrease_button = QtWidgets.QPushButton()
        self.big_step_decrease_button.setIcon(QIcon(':/icons/double_arrow_left'))
        self.big_step_decrease_button.clicked.connect(
            lambda: self.step_slice(vol, "Multi Step Decrease"))
        self.step_decrease_button = QtWidgets.QPushButton()
        self.step_decrease_button.setIcon(QIcon(':/icons/arrow_left'))
        self.step_decrease_button.clicked.connect(
            lambda: self.step_slice(vol, "Single Step Decrease"))
        self.step_increase_button = QtWidgets.QPushButton()
        self.step_increase_button.setIcon(QIcon(':/icons/arrow_right'))
        self.step_increase_button.clicked.connect(
            lambda: self.step_slice(vol, "Single Step Increase"))
        self.big_step_increase_button = QtWidgets.QPushButton()
        self.big_step_increase_button.setIcon(
            QIcon(':/icons/double_arrow_right'))
        self.big_step_increase_button.clicked.connect(
            lambda: self.step_slice(vol, "Multi Step Increase"))
        # Key Slice drop down navigation
        self.key_slice_drop_down = QtWidgets.QComboBox()
        self.key_slice_drop_down.addItem("~")
        self.key_slice_drop_down.activated.connect(
            lambda: self.set_slice(vol, self.key_slice_drop_down.currentText()))

        # adding to slice nav layout
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slice_slider)
        slider_layout.addWidget(self.slice_index)
        # adding step nav to layout
        step_layout = QtWidgets.QHBoxLayout()
        step_layout.addWidget(self.big_step_decrease_button)
        step_layout.addWidget(self.step_decrease_button)
        step_layout.addWidget(self.step_increase_button)
        step_layout.addWidget(self.big_step_increase_button)
        step_layout.addWidget(self.key_slice_drop_down)
        # adding to overall window layout
        slice_layout.addLayout(slider_layout)
        slice_layout.addLayout(step_layout)

        # -----------------------------Tool Bar Layout-------------------------------
        # segmentation loader -------------------------------------------------------
        self.segmentation_list = QtWidgets.QListWidget()
        fill_seg_list(self, vol, seg_dir, self.segmentation_list)
        self.segmentation_list.itemClicked.connect(
            lambda uuid: self.handle_list_click(seg_dir, uuid))
        
        # Segmentation settings -----------------------------------------------------
        # save button
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('Save Points')
        self.save_button.clicked.connect(lambda: self.save_points(vol, seg_dir))

        # view button
        self.view_button = QtWidgets.QPushButton()
        self.view_button.setText('View all points')
        self.view_button.clicked.connect(lambda: self.popup())

        # Show slice shadows toggel 
        self.show_shadows_toggle = QtWidgets.QCheckBox("Show Slice Shadows")
        self.show_shadows_toggle.setChecked(True)
        self.show_shadows_toggle.stateChanged.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        # undo last point button
        self.undo_point_button = QtWidgets.QPushButton()
        self.undo_point_button.setText('Undo Last Point')
        self.undo_point_button.clicked.connect(lambda: self.undo_point(vol))
        # clear slice button
        self.clear_slice_button = QtWidgets.QPushButton()
        self.clear_slice_button.setText('Clear Slice')
        self.clear_slice_button.clicked.connect(lambda: self.clear_slice(vol))
        # clear all button
        #  - first: set a pop window to confirm
        self.clear_all_msg = QtWidgets.QMessageBox()
        self.clear_all_msg.setWindowTitle("Clear All Points")
        self.clear_all_msg.setText(
            "This action cannot be undone, do you want to continue?")
        self.clear_all_msg.setIcon(QMessageBox.Icon.Warning)
        self.clear_all_msg.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes)
        self.clear_all_msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        self.clear_all_msg.buttonClicked.connect(
            lambda i: False if (i.text() == 'Cancel') else self.clear_all(vol))
        #  - second: create button
        self.clear_all_button = QtWidgets.QPushButton()
        self.clear_all_button.setText('Clear All')
        self.clear_all_button.clicked.connect(lambda: self.clear_all_msg.exec())

        # interpolation settings ---------------------------------------------------
        interpolation_options = QtWidgets.QGroupBox()
        interpolation_opt_layout = QtWidgets.QVBoxLayout()
        interpolation_options.setLayout(interpolation_opt_layout)
        nonlinear_settings = QtWidgets.QFrame()
        nonlinear_settings_layout = QtWidgets.QVBoxLayout()
        nonlinear_settings.setLayout(nonlinear_settings_layout)

        # Interpolation type dropdown
        self.interpolation_type_dropdown = QtWidgets.QComboBox()
        self.interpolation_type_dropdown.addItems(["linear", "non-linear"])
        self.interpolation_type_dropdown.activated.connect(
            lambda: self.interpolation_type_dropdown_handler(nonlinear_settings))
        interpolation_type_layout = QtWidgets.QHBoxLayout()
        interpolation_type_layout.addWidget(QtWidgets.QLabel("Interpolation type:"))
        interpolation_type_layout.addWidget(self.interpolation_type_dropdown)

        # Interpolation advanced settings pop up
        self.advanced_settings_button = QtWidgets.QPushButton()
        self.advanced_settings_button.setText('Advanced Settings')
        self.advanced_settings_button.clicked.connect(lambda: self.popup_advanced_settings())

        # Non linear settings layout ----------|
        # Show Canny edges button
        self.show_edges_check = QtWidgets.QCheckBox("Show Canny Edges")
        self.show_edges_check.setChecked(False)
        self.show_edges_check.stateChanged.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        
        # Draw Normals
        self.draw_normals = QtWidgets.QCheckBox("Draw normals")
        self.draw_normals.setChecked(True)
        self.draw_normals.stateChanged.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        
        #Edge thresholds 
        self.edge_threshold1 = QtWidgets.QLineEdit()
        self.edge_threshold1.setMaxLength(5)
        self.edge_threshold1.setText("100")
        self.edge_threshold1.returnPressed.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        edge_threshold1_layout = QtWidgets.QHBoxLayout()
        edge_threshold1_layout.addWidget(QtWidgets.QLabel("Lower edge threshold:"))
        edge_threshold1_layout.addWidget(self.edge_threshold1)
        self.edge_threshold2 = QtWidgets.QLineEdit()
        self.edge_threshold2.setMaxLength(5)
        self.edge_threshold2.setText("120")
        self.edge_threshold2.returnPressed.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
        edge_threshold2_layout = QtWidgets.QHBoxLayout()
        edge_threshold2_layout.addWidget(QtWidgets.QLabel("Higher edge threshold:"))
        edge_threshold2_layout.addWidget(self.edge_threshold2)

        # adding button to layout
        toolbar_layout.addWidget(QtWidgets.QLabel("Previous segmentations"))
        toolbar_layout.addWidget(self.segmentation_list)
        
        segmentation_options = QtWidgets.QGroupBox()
        segmentation_opt_layout = QtWidgets.QVBoxLayout()
        segmentation_options.setLayout(segmentation_opt_layout)
        segmentation_opt_layout.addWidget(self.undo_point_button)
        segmentation_opt_layout.addWidget(self.clear_slice_button)
        segmentation_opt_layout.addWidget(self.clear_all_button)
        segmentation_opt_layout.addWidget(self.show_shadows_toggle)
        toolbar_layout.addWidget(segmentation_options)

        # Interpolation type label-dropdown pair
        interpolation_opt_layout.addLayout(interpolation_type_layout)
        nonlinear_settings_layout.addLayout(edge_threshold1_layout)
        nonlinear_settings_layout.addLayout(edge_threshold2_layout)
        nonlinear_settings_layout.addWidget(self.draw_normals)
        nonlinear_settings_layout.addWidget(self.show_edges_check)
        interpolation_opt_layout.addWidget(nonlinear_settings)
        nonlinear_settings.hide()

        toolbar_layout.addWidget(interpolation_options)
        toolbar_layout.addWidget(self.view_button)
        toolbar_layout.addWidget(self.save_button)

        # Pop window in case number of points is incorrect
        self.incorrect_points = QtWidgets.QMessageBox()
        self.incorrect_points.setWindowTitle("Incorrect number of points")
        self.incorrect_points.setText("You cannot save or interpolate if your "
                                      "slices do not have the same number of "
                                      "points")
        self.incorrect_points.setStandardButtons(
            QMessageBox.StandardButton.Cancel)
        self.incorrect_points.setDefaultButton(
            QMessageBox.StandardButton.Cancel)
        self.incorrect_points.buttonClicked.connect(lambda: False)

        # ---------------------------Variable Storage---------------------------------
        # lines is a dictionary that stores slice -> [list of points (x, y, z)]
        self.lines = dict()
        self.active_line = 0
        self.lines[self.active_line] = dict()
        self.init_x_zoom = self.ax.get_xlim()
        self.init_y_zoom = self.ax.get_ylim()
        self.zoom_width = self.init_x_zoom
        self.zoom_height = self.init_y_zoom
        self.xAxisLim = None
        self.yAxisLim = None
        self.pan_limit = False
        self.global_xlim = self.vol[0].shape[1]
        self.global_ylim = self.vol[0].shape[0]

        # ---------------------------Segmentation Point Drawing---------------------------
        cidClick = self.canvas.mpl_connect('button_press_event', self.onclick)

        cidClick = self.canvas.mpl_connect('button_release_event', self.onrelease)

        # ---------------------------Matplotlib resizeing with keyboard shotcut---------------------------
        cidScroll = self.canvas.mpl_connect('scroll_event', self.onScroll)

    # Function to be called when the mouse is scrolled
    def onScroll(self, event):
        if event.inaxes == self.ax:
            self.toolbar.push_current()
            base_scale = 1.15
            # get the current x and y limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            global_xlim = self.vol[0].shape[1]
            global_ylim = self.vol[0].shape[0]

            zoom_limit = False
            
            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # deal with zoom out
                if (cur_xlim[0] >= 0 and cur_ylim[0] >= 0 and 
                    cur_xlim[1] <= global_xlim and cur_ylim[1] <= global_ylim):
                    scale_factor = base_scale
                else:
                    self.ax.set_xlim(self.init_x_zoom)
                    self.ax.set_ylim(self.init_y_zoom)
                    self.zoom_width = self.init_x_zoom
                    self.zoom_height = self.init_y_zoom
                    zoom_limit = True
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            if (not zoom_limit):
                # set new limits
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                
                relx = (cur_xlim[1]-event.xdata)/(cur_xlim[1]-cur_xlim[0])
                rely = (cur_ylim[1]-event.ydata)/(cur_ylim[1]-cur_ylim[0])

                # Store zoom variables
                self.zoom_width = [event.xdata-new_width*(1-relx), event.xdata+new_width*(relx)]
                self.zoom_height = [event.ydata-new_height*(1-rely), event.ydata+new_height*(rely)]

                self.ax.set_xlim(self.zoom_width)
                self.ax.set_ylim(self.zoom_height)
            
            self.canvas.draw()
            self.toolbar.push_current()


    # draws in the shadows for the key slices
    def draw_shadow(self, line_idx, shadow_color, key_slice):
        for i in range(len(self.lines[line_idx][key_slice]) - 1):
            point = self.lines[line_idx][key_slice][i]
            next_point = self.lines[line_idx][key_slice][i + 1]
            self.ax.plot([point[0], next_point[0]], [point[1], next_point[1]],
                         color=shadow_color, alpha=0.5)
            self.ax.add_artist(
                plt.Circle((point[0], point[1]), 3.5, color=shadow_color,
                           alpha=0.5))
        self.ax.add_artist(plt.Circle((self.lines[line_idx][key_slice][-1][0],
                                       self.lines[line_idx][key_slice][-1][1]),
                                      3.5, color=shadow_color, alpha=0.5))
        self.ax.add_artist(plt.Rectangle((self.lines[line_idx][key_slice][0][
                                              0] - 3.5,
                                          self.lines[line_idx][key_slice][0][
                                              1] - 3.5), 7.5, 7.5,
                                         color=shadow_color, alpha=1,
                                         zorder=50))

    # undos the last point that was drawn on that slice
    def undo_point(self, vol):
        slice_num = self.slice_slider.value()
        #making sure to only delete points from current page
        if slice_num in self.lines[self.active_line]:
            length = len(self.lines[self.active_line][slice_num])
            if (length > 0):
                #remove the last point drawn from the list
                self.lines[self.active_line][slice_num].pop()
                
                #if there are no more points on the slice, remove the keyslice from the list
                if (length == 1):
                    self.lines[self.active_line].pop(slice_num)
                
                self.update_slice(vol, slice_num)


    def insert_ax(self, vol, initial_slice):
        self.ax = self.canvas.figure.subplots()
        self.ax.tick_params(labelcolor='white', colors='white')
        self.ax.imshow(vol[initial_slice])
        self.bar = None 

    # Loads the slice and its points
    def update_slice(self, vol, val):
        self.ax.clear()
        if (self.show_edges_check.isChecked()):
            self.ax.imshow(canny_edge(vol[val], int(self.edge_threshold1.text()), int(self.edge_threshold2.text())), cmap=self.colormap)
        else:
            self.ax.imshow(vol[val])

        # Set the zoom
        self.ax.set_xlim(self.zoom_width)
        self.ax.set_ylim(self.zoom_height)

        # Update the slice index box
        self.slice_index.setText(str(val))

        # For each open segmentation in the list
        for uuid in self.lines:
            lines = self.lines[uuid]

            if uuid == self.active_line:
                circle_size = 7
            else:
                circle_size = 0

            # Checking to see if slice is a key slice
            # if not key slice
            if int(val) not in lines:
                self.key_slice_drop_down.setCurrentText("~")
            else:
                self.key_slice_drop_down.setCurrentText(str(val))

            # loading in the points ghost (preview)
            if self.show_shadows_toggle.isChecked() and len(lines) != 0:
                # putting in shadow for the previous key slice
                last_slice = find_previous_key(int(val),
                                               lines)  # previous key slice shadow
                if last_slice != -1:
                    self.draw_shadow(uuid, 'black', last_slice[0][2])

                # putting in the shadow for the next key slice
                next_slice = find_next_key(int(val),
                                           lines)  # next key slice shadow
                if next_slice != -1:
                    self.draw_shadow(uuid, 'white', next_slice[0][2])

            # loading in the points
            if int(val) in lines:
                for i in range(len(lines[int(val)]) - 1):
                    point = lines[int(val)][i]
                    next_point = lines[int(val)][i + 1]
                    self.ax.plot([point[0], next_point[0]],
                                    [point[1], next_point[1]], color='red')
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), 3.5, color='red'))
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), circle_size,
                                    facecolor='none', edgecolor='red'))

                self.ax.add_artist(
                    plt.Circle((lines[int(val)][-1][0], lines[int(val)][-1][1]),
                                3.5, color='red'))
                self.ax.add_artist(
                    plt.Circle((lines[int(val)][-1][0], lines[int(val)][-1][1]),
                                circle_size, facecolor='none', edgecolor='red'))

            # drawing the interpolated points on slices between keyslices
            elif verify_partial_interpolation(int(val), lines):
                partial_interpolation(self.ax, lines, int(val), 
                                      type=self.interpolation_type_dropdown.currentText(), 
                                      vol=vol,
                                      draw_edges=self.draw_normals.isChecked(),
                                      edge_threshold1=int(self.edge_threshold1.text()), 
                                      edge_threshold2=int(self.edge_threshold2.text()), 
                                      circle_size=circle_size
                                      )

        self.canvas.draw_idle()

    """
    Function to be called when mouse button is released
    This function releases pan then draws a point if the 
    mouse didn't pan and otherwise does nothing
    :param self 
    :param event
    """
    def onrelease(self, event):
        # Release pan and get current xlimit and y limit values
        self.canvas.toolbar.release_pan(event)
        curr_xAxisLim = self.ax.get_xlim()
        curr_yAxisLim = self.ax.get_ylim()

        if event.button == 1: # Left release
            
            # Add a point if the limits have not changed since the press action
            if self.xAxisLim == curr_xAxisLim and self.yAxisLim == curr_yAxisLim:
                if (event.inaxes == self.ax) and (self.canvas.toolbar.mode == ''):
                    slice_num = self.slice_slider.value()
                    new_point = [event.xdata, event.ydata, slice_num]
                    self.lines[self.active_line].setdefault(slice_num, []).append(
                        new_point)

                    # drawing the line between the past and new point
                    if len(self.lines[self.active_line][slice_num]) > 1:
                        prev_point = self.lines[self.active_line][slice_num][-2]
                        self.ax.plot([prev_point[0], new_point[0]],
                                    [prev_point[1], new_point[1]], color='red')

                    self.ax.add_artist(
                        plt.Circle((event.xdata, event.ydata), 3.5, color="red"))
                    self.ax.add_artist(
                        plt.Circle((event.xdata, event.ydata), 7, facecolor='none',
                                edgecolor='red'))
                    self.canvas.draw_idle()

                    # Add tracker on shadows
                    # Previous slice shadow
                    if self.show_shadows_toggle.isChecked() and len(self.lines[self.active_line][slice_num]) > 0:
                        drawn_points = len(self.lines[self.active_line][slice_num]) - 1
                        # putting in shadow for the previous key slice
                        last_slice = find_previous_key(int(slice_num),
                                                    self.lines[self.active_line])  # previous key slice shadow
                        if last_slice != -1:
                            self.ax.add_artist(
                                plt.Circle((last_slice[drawn_points][0], last_slice[drawn_points][1]), 7, facecolor='none',
                                        edgecolor='black'))

                        # putting in the shadow for the next key slice
                        next_slice = find_next_key(int(slice_num),
                                                    self.lines[self.active_line])  # next key slice shadow
                        if next_slice != -1:
                            self.ax.add_artist(
                                plt.Circle((next_slice[drawn_points][0], next_slice[drawn_points][1]), 7, facecolor='none',
                                        edgecolor='white'))

                    # on slice that has point == key slice and add it to the key slice list
                    # Find slice in lines dictionary
                    if slice_num in self.lines[self.active_line]:
                        # if not already on list
                        if len(self.lines[self.active_line][slice_num]) == 1:
                            self.key_slice_drop_down.addItem(str(slice_num))
                            self.key_slice_drop_down.setCurrentText(str(slice_num))
            else: 
                # If pan ends up outside of bounds, move it back in
                self.fixPan(event)

   
    """
    Function to be called when the slice is clicked to either enable the 
    pan function(left click) or select current segmentation(right click)
    :param self 
    :param event
    """
    def onclick(self, event):
        if (event.inaxes == self.ax) and (self.canvas.toolbar.mode == ''):
            self.pan_limit = False

            if event.button == 1: # Left click
                # Save current values for limits
                self.xAxisLim = self.ax.get_xlim()
                self.yAxisLim = self.ax.get_ylim()

                if (self.xAxisLim[0] >= 0 and self.yAxisLim[1] >= 0 and 
                self.xAxisLim[1] <= self.global_xlim and self.yAxisLim[0] <= self.global_ylim):
                    # Enable pan
                    self.canvas.toolbar.press_pan(event)  

            elif event.button == 3:  # Right click
                slice_num = self.slice_slider.value()
                min = 99999999
                closest_line = 0
                for uuid in self.lines:
                    seg = self.lines[uuid]
                    if slice_num in seg:
                        temp_min = find_min(
                            [event.xdata, event.ydata, slice_num],
                            seg[slice_num])
                        if temp_min < min:
                            min = temp_min
                            closest_line = uuid
                self.set_active(closest_line)

    """
    Function that moves the image so the user cannot pan off of the screen
    :param self 
    :param event
    """
    def fixPan(self, event):
        # Save current values
        curr_xAxisLim = self.ax.get_xlim()
        curr_yAxisLim = self.ax.get_ylim()

        # If pan goes off the left side
        if (curr_xAxisLim[0] < 0):

            # Save difference between the side off the screen and the limit to fix the opposing side
            xdif = 0 - curr_xAxisLim[0]

            # Adjust limits
            list_curr_xAxisLim = list(curr_xAxisLim)
            list_curr_xAxisLim[0] = 0
            list_curr_xAxisLim[1] += xdif
            curr_xAxisLim = tuple(list_curr_xAxisLim)

            # Reset limits
            self.ax.set_xlim(curr_xAxisLim)
            self.ax.set_ylim(curr_yAxisLim)

        # If pan goes off the top
        if (curr_yAxisLim[1] < 0):

            # Save difference between the side off the screen and the limit to fix the opposing side
            ydif = 0 - curr_yAxisLim[1]

            # Adjust limits
            list_curr_yAxisLim = list(curr_yAxisLim)
            list_curr_yAxisLim[1] = 0
            list_curr_yAxisLim[0] += ydif
            curr_yAxisLim = tuple(list_curr_yAxisLim)

            # Reset limits
            self.ax.set_xlim(curr_xAxisLim)
            self.ax.set_ylim(curr_yAxisLim)

        # If pan goes off the right side
        if (curr_xAxisLim[1] > self.global_xlim):

            # Save difference between the side off the screen and the limit to fix the opposing side
            xdif = curr_xAxisLim[1] - self.global_xlim

            # Adjust limits
            list_curr_xAxisLim = list(curr_xAxisLim)
            list_curr_xAxisLim[1] = self.global_xlim
            list_curr_xAxisLim[0] -= xdif
            curr_xAxisLim = tuple(list_curr_xAxisLim)

            # Reset limits
            self.ax.set_xlim(curr_xAxisLim)
            self.ax.set_ylim(curr_yAxisLim)

        # If pan goes off the bottom
        if (curr_yAxisLim[0] > self.global_ylim):

            # Save difference between the side off the screen and the limit to fix the opposing side
            ydif = curr_yAxisLim[0] - self.global_ylim

            # Adjust limits
            list_curr_yAxisLim = list(curr_yAxisLim)
            list_curr_yAxisLim[0] = self.global_ylim
            list_curr_yAxisLim[1] -= ydif
            curr_yAxisLim = tuple(list_curr_yAxisLim)

            # Reset limits
            self.ax.set_xlim(curr_xAxisLim)
            self.ax.set_ylim(curr_yAxisLim)
        
        # Redraw canvas
        self.canvas.draw()
        self.toolbar.push_current()

        # Store zoom variables
        self.zoom_width = curr_xAxisLim
        self.zoom_height = curr_yAxisLim


    # function set the slice as active
    def set_active(self, uuid):
        self.active_line = uuid
        # clear the current selection
        self.clear_selected()
        item = self.segmentation_list.findItems(str(uuid),
                                                Qt.MatchFlag.MatchExactly)
        if not item:  # the list is empty meaning the segmentation is not one which was previously saved
            self.segmentation_list.setCurrentItem(
                self.segmentation_list.currentItem(),
                QtCore.QItemSelectionModel.SelectionFlag.Deselect)
        else:
            self.segmentation_list.setCurrentItem(item[0],
                                                  QtCore.QItemSelectionModel.SelectionFlag.Select)

        # need to make sure that the key_slice_drop down matches the active seg
        # clearing the key-slices drop down
        self.key_slice_drop_down.clear()
        self.key_slice_drop_down.addItem("~")
        # filling with the key slices from the dictionary
        for keySlice in self.lines[uuid]:
            self.key_slice_drop_down.addItem(str(keySlice))

        self.update_slice(self.vol, self.slice_slider.value())

    # Deselects any selected items in the list
    def clear_selected(self):
        selected = self.segmentation_list.selectedItems()
        for item in selected:
            self.segmentation_list.setCurrentItem(item,
                                                  QtCore.QItemSelectionModel.SelectionFlag.Deselect)

    # Function that is used to set the current slice using the text input box
    def set_slice(self, vol, val):
        if val == "~":
            self.key_slice_drop_down.setCurrentText(
                str(self.slice_slider.value()))
        else:
            sliceNum = int(val)

            if sliceNum < 0:
                sliceNum = vol.shape[0] + sliceNum
            elif sliceNum > vol.shape[0] - 1:
                sliceNum = sliceNum - vol.shape[0]
            self.slice_index.setText(str(sliceNum))

            self.slice_slider.setValue(sliceNum)

    # Function that is used to navigate the slice by set amounts
    def step_slice(self, vol, type):
        slice_num = self.slice_slider.value()

        # determining which button was clicked and incrementing the slice accourdingly
        # if the double arrows are pressed it will move to the nearest key slice in that direction
        # if no keyslices in that direction the slice will jump 50
        if type == "Multi Step Decrease":
            previous_slice = find_previous_key(slice_num,
                                               self.lines[self.active_line])
            if previous_slice == -1:
                slice_num = slice_num - 50
            else:
                slice_num = previous_slice[0][2]
        elif type == "Single Step Decrease":
            slice_num = slice_num - 1
        elif type == "Single Step Increase":
            slice_num = slice_num + 1
        elif type == "Multi Step Increase":
            next_slice = find_next_key(slice_num, self.lines[self.active_line])
            if next_slice == -1:
                slice_num = slice_num + 50
            else:
                slice_num = next_slice[0][2]

        # Checking for looping out of bounds
        if slice_num < 0:
            slice_num = vol.shape[0] + slice_num
        elif slice_num > vol.shape[0] - 1:
            slice_num = slice_num - vol.shape[0]

        self.slice_slider.setValue(
            slice_num)  # setting slider value and auto calling the update function

    def merge_segmentations(self, vol, seg_dir):
        # TBD

        self.update_slice(vol, self.slice_slider.value())

    def handle_list_click(self, seg_dir, uuid):
        if uuid.checkState() == Qt.CheckState.Checked:
            if not (uuid.text() in self.lines):
                self.load_segmentation(seg_dir, uuid.text())
            self.set_active(uuid.text())
        else:
            self.unload_segmentation(uuid.text())
            self.set_active(self.active_line)

    # finds and returns the text of the selected items
    def find_checked(self):
        checked_list = []
        for i in range(self.segmentation_list.count()):
            item = self.segmentation_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_list.append(item.text())
        return checked_list

    def load_segmentation(self, seg_dir, seg):
        # find all of the segmentations which have their boxes clicked
        if os.path.isfile(os.path.join(seg_dir, seg + '/pointset.json')):
            # Load pointset from JSON file
            self.lines[seg] = load_json(seg_dir, seg)
        elif os.path.isfile(os.path.join(seg_dir, seg + '/pointset.vcps')):
            # Load points from VCPS file
            self.lines[seg] = load_vcps(seg_dir, seg)
        else:
            return

        for key in self.lines[seg]:
            self.key_slice_drop_down.addItem(str(key))
        self.update_slice(self.vol, self.slice_slider.value())

    def unload_segmentation(self, seg):
        if seg in self.lines:
            del self.lines[seg]
            self.set_active(0)
            self.update_slice(self.vol, self.slice_slider.value())

    # Function that save the points out ---------> needs to have the repetition fixed with the code that Bruno added
    def save_points(self, vol, seg_dir):
        if not verify_full_interpolation(self.lines[self.active_line]):
            self.incorrect_points.exec()
            return
        uuid = get_date()
        interpolation = full_interpolation(self.lines[self.active_line])
        write_ordered_vcps(get_segmentation_dir(seg_dir, uuid), interpolation)
        write_metadata(get_segmentation_dir(seg_dir, uuid), uuid)
        write_seg_json(get_segmentation_dir(seg_dir, uuid),
                       self.lines[self.active_line])
        QtWidgets.QListWidgetItem(uuid, self.segmentation_list).setCheckState(
            Qt.CheckState.Checked)

        if (Path(str(str(seg_dir) + "/fromInterpolator")).is_dir()):
            write_ordered_vcps(str(str(seg_dir) + "/fromInterpolator"), interpolation)
            write_metadata(str(str(seg_dir) + "/fromInterpolator"), "fromInterpolator")
        print("Points saved out")

    def clear_slice(self, vol):
        key = self.slice_slider.value()
        if key in self.lines[self.active_line].keys():
            # deletes the dictionary slice along with its points
            del self.lines[self.active_line][key]
        else:
            print("Slice already empty")

        # Clear slice from key slices bar
        self.key_slice_drop_down.setCurrentText("~")
        # find the index of the slice to be removed in the key slice dropdown list
        for i in range(1, self.key_slice_drop_down.count()):
            if str(key) == self.key_slice_drop_down.itemText(i):
                self.key_slice_drop_down.removeItem(i)

        self.update_slice(vol, self.slice_slider.value())

    def clear_all(self, vol):
        # deletes the dictionary slice along with its points
        self.lines[self.active_line].clear()

        # clearing the key-slices drop down
        self.key_slice_drop_down.clear()
        self.key_slice_drop_down.addItem("~")

        self.update_slice(vol, self.slice_slider.value())
        return True
    
    def interpolation_type_dropdown_handler(self, item):
        if(self.interpolation_type_dropdown.currentText() == 'linear'):
            item.hide()
        else:
            item.show()
        self.update_slice(self.vol, self.slice_slider.value())

    
    def popup(self):
        popup = ViewPopUp(self.vol, self.lines[self.active_line])
        popup.exec()
        


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
    window = MainWindow(vol, segmentation_dir)
    window.show()
    # allows for exit from the application
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')


if __name__ == "__main__":
    main()
