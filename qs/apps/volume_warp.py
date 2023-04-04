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

import qs.apps.quick_segment

from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_interpolation)
from qs.math import (find_min, calculate_sq_distance)

# -------------------------------------------------------------------
#                             SEGMENT WINDOW CLASS
# -------------------------------------------------------------------
class WarpWindow(QtWidgets.QWidget):
    ax = None
    bar = None

    def __init__(self, vol, seg_dir, initial_slice):
        super().__init__()

        # -------------initial window specs---------------
        self.vol = vol

        # ------------------------------Volume Warping Window GUI-----------------------------
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
        #------buttons related to warping------^

        # save button
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('Save Points')
        self.save_button.clicked.connect(lambda: self.save_points(vol, seg_dir))
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
        #  - second: create button to clear all points
        self.clear_all_button = QtWidgets.QPushButton()
        self.clear_all_button.setText('Clear All')
        self.clear_all_button.clicked.connect(lambda: self.clear_all_msg.exec())
        
        # adding button to layout
        toolbar_layout.addWidget(QtWidgets.QLabel("Previous segmentations"))
        toolbar_layout.addWidget(self.segmentation_list)

        toolbar_layout.addWidget(self.warp_button)
        toolbar_layout.addWidget(self.unwarp_button)

        toolbar_layout.addWidget(self.undo_point_button)
        toolbar_layout.addWidget(self.clear_slice_button)
        toolbar_layout.addWidget(self.clear_all_button)

        toolbar_layout.addWidget(self.save_button)

        # ---------------------------Variable Storage---------------------------------
        # lines is a dictionary that stores slice -> [list of points (x, y, z)]
        self.lines = dict()
        self.active_line = 0
        self.lines[self.active_line] = dict()
        self.init_x_zoom = self.ax.get_xlim()
        self.init_y_zoom = self.ax.get_ylim()
        self.zoom_width = self.init_x_zoom
        self.zoom_height = self.init_y_zoom

        # #storage of moved points for warp
        self.active_point = ([-1, -1, -1, -1])
        self.gridSize = 10
        self.ogPointLoc = ([]) #list that will hold all of the og point locations
        self.newPointLoc = ([]) #list that will hold all of the new point locations
        

        # ---------------------------Segmentation Point Drawing---------------------------
        cidClick = self.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event, vol))

        # ---------------------------Matplotlib resizeing with keyboard shotcut------------
        cidScroll = self.canvas.mpl_connect('scroll_event', self.onScroll)

        # --------------------Adding in image grid for points----------------v
        self.draw_point_grid(vol, True)


    #--------------------------Image Warping----------------------v
    def draw_point_grid(self, vol, first):
        """
        Draws the points that will be used to warp the image on the screen 
        @param Volume
        """
        height = int(vol.shape_y)
        width = int(vol.shape_x)
        count = 0

        #image size 695x551 (WxH)
        for i in range(0, width+1, int(width/self.gridSize)):
            for j in range(0, height+1, int(height/self.gridSize)):
                #filling array but only for the first time
                if first:
                    self.ax.add_artist(
                            plt.Circle([i, j], 3.5, color="yellow")
                        )
                        
                    self.ogPointLoc.append([i,j])
                    self.newPointLoc.append([-1,-1])
                    
                else:
                   
                    # if the point has been moved
                    if self.newPointLoc[count][0] != -1:
                        self.ax.add_artist(plt.Circle([self.newPointLoc[count][0], self.newPointLoc[count][1]], 3.5, color="blue"))
                    else:
                        self.ax.add_artist(plt.Circle([self.ogPointLoc[count][0], self.ogPointLoc[count][1]], 3.5, color="yellow"))

                count = count + 1

        self.canvas.draw_idle()

   
    def warp(self, vol, seg_dir):
        """
        Warps the image based on the change in point positions
        @param Volume
        @param segmentation directory 
        """
        print("WAaAaaaARrrrRRppPppIIiiiIInnnnNNNNnnnGGGGgGGGgggggg")

        #clean the arrays up to prevent -1's
        count = 0
        for point in self.newPointLoc: 
            if point[0] == -1:
                point[0] = self.ogPointLoc[count][0]
                point[1] = self.ogPointLoc[count][1]
            count = count + 1
        #convert to Numpy array
        ogNParray = np.array(self.ogPointLoc, np.float32)
        newNParray = np.array(self.newPointLoc, np.float32)

        matrix = cv2.getPerspectiveTransform(ogNParray, newNParray) #-----> current problem is that the warp function is for 4 points only 

        warpedImage = cv2.warpPerspective(vol[self.slice_slider.value()], matrix, (695, 551))
        cv2.imshow("og Image", vol[self.slice_slider.value()])
        cv2.imshow("warped image", warpedImage)

        print("Done with WAaAaaaARrrrRRppPpp")

    
    def unwarp(self, vol, seg_dir):
        """
        Unwarps the previous warp to return image to og state
        @param Volume
        @param segmentation directory 
        """
        print("UNDO WAaAaaaARrrrRRppPpp")

    
    def getBounds(self, increment, pos):
        """
        Gets the upper and lower bounds for a given points posistion 
        :param self
        :param increment
        :param pos
        :returns lower and upper bounds
        """
        mult = pos - (pos % increment)
        return [mult, mult+increment]
    
   
    def getClosestPoint(self, points, pos):
        """
        Gets the point that is the closest to the 4 surround points provided 
        :param self
        :param points
        :param pos
        :returns closest point position
        """
        min_dist = ([pos, 999999999999]) #big number to compare against 
        for point in points:
            dist = calculate_sq_distance(pos, point)
            if dist <= min_dist[1]:
                min_dist[0] = point
                min_dist[1] = dist
        return min_dist[0]


    #--------------------------Image Warping----------------------^

    #----------------------------Mouse Functions----------------------------v
    # Function to be called when the mouse is scrolled (zoom)
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
    def onclick(self, event, vol):
        if (event.inaxes == self.ax) and (self.canvas.toolbar.mode == ''):
            if event.button == 1:  # Left click
                if(self.active_point[0] == -1):
                    print("You must choose a point to move first")
                else:
                    print("Moving Point: ", self.active_point)

                    #moving point to where the person clicked
                    point = ([event.xdata, event.ydata])
                    
                    #moving choosen point and saving new location in newPointLoc
                    self.set_active(1, point)

            
            elif event.button == 3:  # Right click
                slice_num = self.slice_slider.value()

                xbounds = self.getBounds(int(vol.shape_x/self.gridSize), event.xdata)
                ybounds = self.getBounds(int(vol.shape_y/self.gridSize), event.ydata)

                surrounding_points = ([ [xbounds[0], ybounds[0]], [xbounds[1],ybounds[0]], [xbounds[0], ybounds[1]], [xbounds[1], ybounds[1]] ])
                
                closest_point = self.getClosestPoint(surrounding_points, [event.xdata, event.ydata])
                
                #set the point as active and save loaction in the pointLoc
                self.set_active(0, closest_point) #og point position
                # self.set_active(1, [self.active_point[0], self.active_point[1]]) #moved point position

                self.canvas.draw_idle()

    #--------------------------------matplotlib GUI----------------------------v
    def insert_ax(self, vol, initial_slice):
        self.ax = self.canvas.figure.subplots()
        self.ax.tick_params(labelcolor='white', colors='white')
        self.ax.imshow(vol[initial_slice])
        self.bar = None 

    # Loads the slice and its points
    def update_slice(self, vol, val):
        self.ax.clear()
        self.ax.imshow(vol[val])

        #----------------warping related--------------
        #draw point grid
        self.draw_point_grid(vol, False)

        #if there is a ligit active point make it visably active
        if self.active_point[0] != -1:
            self.ax.add_artist(
                plt.Circle([self.active_point[2], self.active_point[3]], 3.5, color='red'))
            self.ax.add_artist(
                plt.Circle([self.active_point[2], self.active_point[3]], 7,
                    facecolor='none', edgecolor='red'))
        
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
    #--------------------------------matplotlib GUI--------------------------^

    

    #-----------------------------Segmentation Line Loading/Switching-----------------------
    # function set the point as active
    def set_active(self, pos, point):
        #saving active point info
        if pos == 0:
            self.active_point[0] = point[0]
            self.active_point[1] = point[1]
           
            # checking if previously moved 
            index = self.ogPointLoc.index([point[0], point[1]]) #finding location in og point array
            if self.newPointLoc[index][0] == -1:
                self.newPointLoc[index][0] = point[0]
                self.newPointLoc[index][1] = point[1]
            
            self.active_point[2] = self.newPointLoc[index][0]
            self.active_point[3] = self.newPointLoc[index][1]

        #moved point loc
        elif pos == 1:
            self.active_point[2] = point[0]
            self.active_point[3] = point[1]

            #updating pointLoc info
            index = self.ogPointLoc.index([self.active_point[0], self.active_point[1]]) #finding location in og point array
            self.newPointLoc[index][0] = point[0]
            self.newPointLoc[index][1] = point[1]
            # print("found'em")
            # print("Points List: ", self.newPointLoc[index])
            # print("Active Point: ", self.active_point)


        self.update_slice(self.vol, self.slice_slider.value())

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

    #------------------------Button Functions-------------------------------v
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