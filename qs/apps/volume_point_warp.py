#Volume Warp Page within the GUI
from __future__ import annotations

import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path

import cv2 #open cv used for warping 

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QMessageBox, QMenuBar, QMenu
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

import qs.apps.quick_segment #import (StackedWindow)

from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_interpolation)
from qs.math import find_min

# -------------------------------------------------------------------
#                             VOLUME WARP WINDOW CLASS
# -------------------------------------------------------------------
class VolumeWarpWindow(QtWidgets.QWidget):
    ax = None
    bar = None


    def __init__(self, vol, seg_dir, initial_slice):
        super().__init__()

         # -------------initial window specs---------------
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
            plt.Figure(figsize=[8,7], layout= 'tight', facecolor='#3d3d3d'))
        # plt.tight_layout()
        
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
        # segmentation loader
        self.segmentation_list = QtWidgets.QListWidget()
        fill_seg_list(self, vol, seg_dir, self.segmentation_list)
        self.segmentation_list.itemClicked.connect(
            lambda uuid: self.handle_list_click(seg_dir, uuid))


        #------buttons related to warping------v
        # warp button
        self.warp_button = QtWidgets.QPushButton()
        self.warp_button.setText('Warp')
        self.warp_button.clicked.connect(lambda: self.warp(vol, seg_dir))
        # unwarp button
        self.unwarp_button = QtWidgets.QPushButton()
        self.unwarp_button.setText('Unwarp')
        self.unwarp_button.clicked.connect(lambda: self.unwarp(vol, seg_dir))

        # Set points button 
        self.set_points_button = QtWidgets.QPushButton()
        self.set_points_button.setText('Set Start Points')
        self.set_points_button.clicked.connect(lambda: self.set_og_points(vol, seg_dir))
        
        # Adding New Points toggel 
        self.new_points_toggle = QtWidgets.QCheckBox("Adding new Points")
        self.new_points_toggle.setChecked(False)
        self.new_points_toggle.stateChanged.connect(lambda: self.flip_value(vol, "Set Points"))

        #------buttons related to warping------^
        

        # save button
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('Save Points')
        self.save_button.clicked.connect(lambda: self.save_points(vol, seg_dir))
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
        
        # adding button to layout
        toolbar_layout.addWidget(QtWidgets.QLabel("Previous segmentations"))
        toolbar_layout.addWidget(self.segmentation_list)

        #WARPING BUTTONS ----v
        toolbar_layout.addWidget(self.warp_button)
        toolbar_layout.addWidget(self.unwarp_button)

        toolbar_layout.addWidget(self.set_points_button)
        toolbar_layout.addWidget(self.new_points_toggle)
        #WARPING BUTTONS -----^

        toolbar_layout.addWidget(self.undo_point_button)
        toolbar_layout.addWidget(self.clear_slice_button)
        toolbar_layout.addWidget(self.clear_all_button)

        toolbar_layout.addWidget(self.show_shadows_toggle)

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

        #-------Storage for the warp--------
        #list of points used for warping the slice
        self.ogWarpPoints = dict() #first 4 contain the boarder points
        self.newWarpPoints = dict()
        self.gridSize = 5
        self.grid_point_count = 0

        self.settingNewPoints = False

        # ---------------------------Segmentation Point Drawing---------------------------
        cidClick = self.canvas.mpl_connect('button_press_event', self.onclick)

        # ---------------------------Matplotlib resizeing with keyboard shotcut---------------------------
        cidScroll = self.canvas.mpl_connect('scroll_event', self.onScroll)

    #-------------------------------Warp Functions------------------------------
    def warp(self, vol, seg_dir):
        """
        Warps the image based on the change in point positions
        @param Volume
        @param segmentation directory 
        """
        print("WAaAaaaARrrrRRppPppIIiiiIInnnnNNNNnnnGGGGgGGGgggggg")

        slice_num = self.slice_slider.value()

        #convert to Numpy array
        ogNP = np.array(self.ogWarpPoints[slice_num], np.int32)
        newNP = np.array(self.newWarpPoints[slice_num], np.int32)
        
        og_img = vol[slice_num]
        #Apply the transform to the image
        new_img = self.transform(og_img, ogNP, newNP)

        #show
        cv2.imshow("Transformed", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Done with WAaAaaaARrrrRRppPpp")

    def transform(self, og_img, og_points, new_points):
        """
        Transforms the og image into the new image
        @param og_img
        @param og_points
        @param new_points
        """
        print("Transform has started")

        new_img = og_img.copy()    

        for set in self.get_triangle_set(og_points):

            #Get the triangles 
            og_tri = og_points[set]
            new_tri = new_points[set]

            #get rid of everything around the triangle with a crop
            og_tri_crop, og_img_crop = self.crop_triangle(og_img, og_tri)
            new_tri_crop, new_img_crop = self.crop_triangle(new_img, new_tri)

            #get the affine transform matrix
            matrix = cv2.getAffineTransform(np.float32(og_tri_crop), np.float32(new_tri_crop))

            #warp
            new_img_warp = cv2.warpAffine(og_img_crop, matrix, (new_img_crop.shape[1], new_img_crop.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            #create triangle mask
            mask = np.zeros(new_img_crop.shape, dtype = np.uint8)
            cv2.fillConvexPoly(mask, np.int32(new_tri_crop), (1.0, 1.0, 1.0), 16, 0)

            #Remove pixels that are already there
            new_img_crop *= 1-mask
            #Add pixels back in masked area
            new_img_crop += new_img_warp*mask

        print("Successfully Transformed")
        return new_img

    def get_triangle_set(self, points):
        """
        Gets the indices triples for every triangle
        @param points
        """        
        #bound rectangle
        bound_rect = (0, 0, (points[0][0]+1), (points[1][1]+1))

        #triangulate the points
        subDiv = cv2.Subdiv2D(bound_rect)
        print(points)
        print(bound_rect)
        for p in points:
            point = (int(p[0]), int(p[1]))
            subDiv.insert(point)

        #Go over all tri's
        for x1, y1, x2, y2, x3, y3 in subDiv.getTriangleList():
            #get index of all of the points
            yield [(points==point).all(axis=1).nonzero()[0][0] for point in [(x1,y1), (x2,y2), (x3,y3)]]


    def crop_triangle(self, img, tri):
        """
        Crops the image so that it only contains the triangle
        @param img
        @param tri
        @return tri_crop, img_crop
        """

        #get the bounds
        bound_rect = cv2.boundingRect(tri)

        #crop the image to bounds
        img_crop = img[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]

        #put triangle in the non cropped area
        tri_crop = [(edge[0]-bound_rect[0], edge[1]-bound_rect[1]) for edge in tri]

        #return the cropped triangle and the cropped image
        return tri_crop, img_crop


    def unwarp(self, vol, seg_dir):
        """
        Unwarps the previous warp to return image to og state
        @param Volume
        @param segmentation directory 
        """
        print("UNDO WAaAaaaARrrrRRppPpp")

    
    def set_og_points(self, vol, seg_dir):
        """
        Saves the original points and adds boarder points to the point set for future warpping 
        @param Volume
        @param segmentation directory 
        """
        slice_num = self.slice_slider.value()
        width = vol.shape[2]
        height = vol.shape[1]

        #Add all of the boarder points as the first 4 points in the ogPoints list
        boarder_points = ([[width,0], [0,height], [width,height]]) #not including zerp bc grid includes it
        self.ogWarpPoints.setdefault(slice_num, []).extend(boarder_points) 
        grid_points = ([])
        for i in range(0, width+1, int(width/self.gridSize)):
            for j in range(0, height+1, int(height/self.gridSize)):
                grid_points.append([i,j])
                self.grid_point_count += 1
        self.ogWarpPoints[slice_num].extend(grid_points)

        #adding each point as (x,y) to the list
        for point in self.lines[self.active_line][slice_num]:
            self.ogWarpPoints[slice_num].append([point[0], point[1]])
        print("OG Points saved to the dict - list")

        #setting up the new points dict so its ready for the new points
        self.newWarpPoints.setdefault(slice_num, []).extend(boarder_points) 
        self.newWarpPoints[slice_num].extend(grid_points)
        
        print("new Points dict - list started")


    def flip_value(self, vol, name):
        """
        Flips the boolean value of the variable given by the param name
        @param name -> the case for what value to be flipped
        """
        if name == "Set Points":
            self.settingNewPoints = not self.settingNewPoints

        self.update_slice(vol, self.slice_slider.value())
        




    #-------------------------------Mouse Functions-------------------------------v
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

    # Function to be called when the slice is clicked to add points
    def onclick(self, event):
        if (event.inaxes == self.ax) and (self.canvas.toolbar.mode == ''):
            if event.button == 1:  # Left click
                slice_num = self.slice_slider.value()
                #checking if placing new points for segmentation 
                if self.settingNewPoints:
                    fullPoints = False #checking to see if the user has put the same number of points as there is in the og line
                    if len(self.newWarpPoints[slice_num]) == len(self.ogWarpPoints[slice_num]):
                        fullPoints = True

                    #ignore all other unput after 8 points are drawn
                    if not fullPoints:
                        new_point = [event.xdata, event.ydata]
                        self.newWarpPoints.setdefault(slice_num, []).append(new_point)

                        # drawing the line between the past and new point
                        if len(self.newWarpPoints[slice_num]) > 5:
                            prev_point = self.newWarpPoints[slice_num][-2] #finding second to last point 
                            self.ax.plot([prev_point[0], new_point[0]],
                                        [prev_point[1], new_point[1]], color='blue')
                            
                        self.ax.add_artist(
                            plt.Circle((event.xdata, event.ydata), 3.5, color="blue"))
                        self.ax.add_artist(
                            plt.Circle((event.xdata, event.ydata), 7, facecolor='none',
                                    edgecolor='blue'))
                        
                    #clicks after the 8 points are placed ignored
                    else:
                        print("You have already placed all of the points that you need")
                        
                    self.canvas.draw_idle()

                #normal point addition
                else:
                    # on slice that has point == key slice and add it to the key slice list
                    # Find slice in lines dictionary
                    if slice_num in self.lines[self.active_line]:
                        # if not already on list
                        if len(self.lines[self.active_line][slice_num]) == 1:
                            self.key_slice_drop_down.addItem(str(slice_num))
                            self.key_slice_drop_down.setCurrentText(str(slice_num))
                    
                    

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

    #---------------------------Matplotlib Functions-----------------------------
    def insert_ax(self, vol, initial_slice):
        self.ax = self.canvas.figure.subplots()
        self.ax.tick_params(labelcolor='white', colors='white')
        self.ax.imshow(vol[initial_slice])
        self.bar = None 

    # Loads the slice and its points
    def update_slice(self, vol, val):
        self.ax.clear()
        self.ax.imshow(vol[val])

        self.ax.set_xlim(self.zoom_width)
        self.ax.set_ylim(self.zoom_height)

        # update the slice index box
        self.slice_index.setText(str(val))

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


            # # loading in the points ghost (preview)
            # if self.show_shadows_toggle.isChecked() and len(lines) != 0:
            #     # putting in shadow for the previous key slice
            #     last_slice = find_previous_key(int(val),
            #                                    lines)  # previous key slice shadow
            #     if last_slice != -1:
            #         self.draw_shadow(uuid, 'black', last_slice[0][2])

            #     # putting in the shadow for the next key slice
            #     next_slice = find_next_key(int(val),
            #                                lines)  # next key slice shadow
            #     if next_slice != -1:
            #         self.draw_shadow(uuid, 'white', next_slice[0][2])

           
           
            #-----------Loading in the Warping points--------------
            alpha_value = 1
            #if we are focusing on the new point locations for the warpping
            if self.settingNewPoints:
                #printing out the new points for warp
                if int(val) in self.newWarpPoints:
                    for i in range((3+self.grid_point_count), len(self.newWarpPoints[int(val)])-1): #not including the 4 boarder points
                        point = self.newWarpPoints[int(val)][i]
                        next_point = self.newWarpPoints[int(val)][i + 1]
                        self.ax.plot([point[0], next_point[0]],
                                        [point[1], next_point[1]], color='blue', alpha = alpha_value)
                        self.ax.add_artist(
                            plt.Circle((point[0], point[1]), 3.5, color='blue',  alpha = alpha_value))
                        self.ax.add_artist(
                            plt.Circle((point[0], point[1]), circle_size,
                                        facecolor='none', edgecolor='blue',  alpha = alpha_value))
                    if len(self.newWarpPoints[int(val)]) > (3+self.grid_point_count):
                        self.ax.add_artist(
                            plt.Circle((self.newWarpPoints[int(val)][-1][0], self.newWarpPoints[int(val)][-1][1]),
                                        3.5, color='blue',  alpha = alpha_value))
                        self.ax.add_artist(
                            plt.Circle((self.newWarpPoints[int(val)][-1][0], self.newWarpPoints[int(val)][-1][1]),
                                        circle_size, facecolor='none', edgecolor='blue',  alpha = alpha_value))
                
                #making og points slightly see through 
                alpha_value = 0.5

            # loading in the points from the og points dict
            if int(val) in lines:
                for i in range(len(lines[int(val)]) - 1):
                    point = lines[int(val)][i]
                    next_point = lines[int(val)][i + 1]
                    self.ax.plot([point[0], next_point[0]],
                                    [point[1], next_point[1]], color='red', alpha = alpha_value)
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), 3.5, color='red',  alpha = alpha_value))
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), circle_size,
                                    facecolor='none', edgecolor='red',  alpha = alpha_value))
                self.ax.add_artist(
                    plt.Circle((lines[int(val)][-1][0], lines[int(val)][-1][1]),
                                3.5, color='red',  alpha = alpha_value))
                self.ax.add_artist(
                    plt.Circle((lines[int(val)][-1][0], lines[int(val)][-1][1]),
                                circle_size, facecolor='none', edgecolor='red',  alpha = alpha_value))

            # drawing the interpolated points on slices between keyslices
            elif verify_interpolation(int(val), lines):
                previous_key = find_previous_key(int(val), lines)
                next_key = find_next_key(int(val), lines)
                point = interpolate_point(int(val), previous_key[0],
                                          next_key[0])
                for i in range(0, len(previous_key) - 1):
                    next_point = interpolate_point(int(val),
                                                   previous_key[i + 1],
                                                   next_key[i + 1])
                    self.ax.plot([point[0], next_point[0]],
                                 [point[1], next_point[1]], color='yellow')
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), 3.5, color='yellow'))
                    self.ax.add_artist(
                        plt.Circle((point[0], point[1]), circle_size,
                                   facecolor='none', edgecolor='yellow'))
                    point = next_point

                last_point = interpolate_point(int(val), previous_key[-1],
                                               next_key[-1])
                self.ax.add_artist(
                    plt.Circle((last_point[0], last_point[1]), 3.5,
                               color='yellow'))
                self.ax.add_artist(
                    plt.Circle((last_point[0], last_point[1]), circle_size,
                               facecolor='none', edgecolor='yellow'))

        self.canvas.draw_idle()

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


    #---------------------------Segemntation Loading/Shadows/Switching-----------------------v
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

    #-------------------------Button Functions--------------------------------

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

        #------clearing new warping points------
        self.newWarpPoints.clear()

        # Clear slice from key slices bar
        self.key_slice_drop_down.setCurrentText("~")
        # find the index of the slice to be removed in the key slice dropdown list
        for i in range(1, self.key_slice_drop_down.count()):
            if str(key) == self.key_slice_drop_down.itemText(i):
                self.key_slice_drop_down.removeItem(i)

        self.update_slice(vol, self.slice_slider.value())

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

        #--------undoing warping points------------
        if len(self.newWarpPoints[slice_num]) > 4: #not undoing the boarder points
            self.newWarpPoints[slice_num].pop()


        self.update_slice(vol, slice_num)

    def clear_all(self, vol):
        # deletes the dictionary slice along with its points
        self.lines[self.active_line].clear()

        # clearing the key-slices drop down
        self.key_slice_drop_down.clear()
        self.key_slice_drop_down.addItem("~")

        self.update_slice(vol, self.slice_slider.value())
        return True
    
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