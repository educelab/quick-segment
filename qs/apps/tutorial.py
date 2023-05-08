from __future__ import annotations

from typing import Union, Tuple

from PyQt6 import QtGui
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QDialog, QLabel, QPushButton,
                             QTabWidget, QVBoxLayout, QWidget)

# noinspection PyUnresolvedReferences
import qs.resources


def _create_video_widget(url: Union[QUrl, str]) -> Tuple[
    QVBoxLayout, QVideoWidget, QMediaPlayer]:
    if not isinstance(url, QUrl):
        url = QUrl(url)

    layout = QVBoxLayout()
    layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    vid_widget = QVideoWidget()
    player = QMediaPlayer()
    player.setSource(url)
    player.setVideoOutput(vid_widget)
    player.setLoops(QMediaPlayer.Loops.Infinite)
    player.play()

    layout.addWidget(vid_widget)

    return layout, vid_widget, player


class TutorialWindow(QDialog):
    ax = None
    bar = None

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # setup window
        self.setMinimumSize(800, 600)
        self.setMaximumSize(960, 720)
        self.setWindowTitle("Tutorials")
        self.setLayout(QVBoxLayout())

        # setup big font
        big_font = QtGui.QFont()
        big_font.setPixelSize(20)
        big_font.setBold(True)

        # create tabs widget
        self.tabs = QTabWidget()
        self.layout().addWidget(self.tabs)

        # keep a reference to all video players
        self.video_players = {}

        # page 1: adding points
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Points")

        page_title = QLabel("Adding Points")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(" - Left click on the canvas to add points")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/click.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['click'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 2: moving points
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Adjust")

        page_title = QLabel("Moving Points")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Left click on a point then drag and drop it where you want to "
            "move it to\n\n"
            "Note: A point selected to move turns blue")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/move.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['move'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 3: Navigation
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Canvas Navigation")

        page_title = QLabel("Canvas Navigation")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Scrolling zooms in and out of the canvas\n"
            " - Left click and drag allows you to pan")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/pan-zoom.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['pan-zoom'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 4: Slice navigation
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Slice Navigation")

        page_title = QLabel("Slice Navigation")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Single arrows move a single slice forward and backwards\n"
            " - Double arrows move to the nearest key slice in the given "
            "direction when no key slice the number of slices indicated by the jump size box\n"
            " - The dropdown allows for viewing all of the key slices and navigate to them")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/slice-nav.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['slice-nav'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 5: Shadows
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Shadows")

        page_title = QLabel("Shadow Segmentations")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - A shadow segmentation shows your points from another key slice\n"
            "   - Black shadows are for prior key slices \n"
            "   - White shadows are for the succeeding key slices\n"
            " - Circles surrounding the points on the shadow help you keep "
            "track of how many points you have placed")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/shadow-seg.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['shadow-seg'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 6: Interpolation
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Interpolation")

        page_title = QLabel("Interpolation")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Between 2 red key slices there are interpolated segmentation "
            "lines represented by the yellow color")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/interpolation.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['interpolation'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # page 7: Segmentation loading
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Load Segmentations")

        page_title = QLabel("Loading and Editing Segmentations")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Previous segmentations can be viewed and edited by selecting on "
            "them in the previous segmentations box\n"
            " - Right click on segmentation select it for editing, indicated "
            "by extra circle\n"
            " - Add your current segmentation to the list by saving points\n\n"
            "Note: It is strongly recommended to load segmentations completed "
            "through the Quick Segment tool. Segmentations from other tools "
            "may not be compatible.")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/load-seg.mp4')
        vid_widget.setMaximumSize(600, 450)
        self.video_players['load-seg'] = vid_player
        page_widget.layout().addLayout(vid_layout)

        # ----button to easly move to next page?------- -> not in use yet
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(lambda: self.next_tut_page())

    def next_tut_page(self):
        self.tabs.setCurrentIndex(self.p2_index)
