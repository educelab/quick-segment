from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

# noinspection PyUnresolvedReferences
import qs.resources
from qs.data import (Volume, fill_seg_list, get_date, get_segmentation_dir,
                     load_json, load_vcps, write_metadata, write_ordered_vcps,
                     write_seg_json)
from qs.interpolation import (find_next_key, find_previous_key,
                              full_interpolation, interpolate_point,
                              verify_full_interpolation, verify_interpolation)
from qs.math import find_min


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
            plt.Figure(figsize=(15, 16), facecolor='#3d3d3d'))

        self.toolbar = NavigationToolbar(self.canvas, self) 
        #removing unnecessary buttons  
        unwanted_buttons = ['Save', 'Subplots', 'Customize']
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

        # Segmentation load button
        # Seg_layout = QtWidgets.QHBoxLayout()
        # self.seg_merge = QtWidgets.QPushButton('Merge')
        # self.seg_merge.clicked.connect(lambda: self.merge_segmentations(vol, seg_dir))
        # Seg_layout.addWidget(self.seg_load_button)
        # Seg_layout.addWidget(self.seg_merge)

        # save button
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('Save Points')
        self.save_button.clicked.connect(lambda: self.save_points(vol, seg_dir))
        # Show slice shadows toggel 
        self.show_shadows_toggle = QtWidgets.QCheckBox("Show Slice Shadows")
        self.show_shadows_toggle.setChecked(True)
        self.show_shadows_toggle.stateChanged.connect(
            lambda: self.update_slice(vol, self.slice_slider.value()))
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
        # ToolBar_Layout.addLayout(Seg_layout)
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

        # ---------------------------Segmentation Point Drawing---------------------------
        cidClick = self.canvas.mpl_connect('button_press_event', self.onclick)

        # ---------------------------Matplotlib resizeing with keyboard shotcut---------------------------
        cidScroll = self.canvas.mpl_connect('scroll_event', self.onScroll)

    # Function to be called when the mouse is scrolled
    def onScroll(self, event):
        if event.inaxes == self.ax:
            self.toolbar.push_current()
            base_scale = 1.2
            # get the current x and y limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)
            # set new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1]-event.xdata)/(cur_xlim[1]-cur_xlim[0])
            rely = (cur_ylim[1]-event.ydata)/(cur_ylim[1]-cur_ylim[0])

            self.ax.set_xlim([event.xdata-new_width*(1-relx), event.xdata+new_width*(relx)])
            self.ax.set_ylim([event.ydata-new_height*(1-rely), event.ydata+new_height*(rely)])
            
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

    def insert_ax(self, vol, initial_slice):
        self.ax = self.canvas.figure.subplots()
        self.ax.tick_params(labelcolor='white', colors='white')
        self.ax.imshow(vol[initial_slice])
        self.bar = None 

    # Loads the slice and its points
    def update_slice(self, vol, val):
        self.ax.clear()
        self.ax.imshow(vol[val])

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

    # Function to be called when the slice is clicked to add points
    def onclick(self, event):
        if (event.inaxes == self.ax) and (self.canvas.toolbar.mode == ''):
            if event.button == 1:  # Left click
                slice_num = self.slice_slider.value()
                new_point = [event.xdata, event.ydata, slice_num]
                self.lines[self.active_line].setdefault(slice_num, []).append(
                    new_point)

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
