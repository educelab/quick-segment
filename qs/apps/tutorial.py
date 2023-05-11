from __future__ import annotations

from typing import Union, Tuple

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QPushButton, QTabWidget, QVBoxLayout, QWidget)

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
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # setup window
        self.setMinimumSize(800, 800)
        self.setMaximumSize(960, 800)
        self.setWindowTitle("Tutorials")
        self.setLayout(QVBoxLayout())

        # setup big font
        big_font = QFont()
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
        self.tabs.addTab(page_widget, "Create")

        page_title = QLabel("Add points")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(" - Left click on the canvas to add points")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/click.mp4')
        self.video_players['click'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # page 2: moving points
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Adjust")

        page_title = QLabel("Adjust points")
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
        self.video_players['move'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # page 3: Navigation
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Canvas navigation")

        page_title = QLabel("Canvas navigation")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Scrolling zooms in and out of the canvas\n"
            " - Left click and drag allows you to pan")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/pan-zoom.mp4')
        self.video_players['pan-zoom'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # page 4: Slice navigation
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Volume navigation")

        page_title = QLabel("Volume navigation")
        page_title.setFont(big_font)
        page_layout.addWidget(page_title)
        page_desc = QLabel(
            " - Single arrows move a single slice forward and backwards\n"
            " - Double arrows move to the nearest key slice in the given "
            "direction when no key slice the number of slices indicated by the "
            "jump size box\n"
            " - The dropdown allows for viewing all of the key slices and "
            "navigate to them")
        page_desc.setWordWrap(True)
        page_layout.addWidget(page_desc)

        vid_layout, vid_widget, vid_player = _create_video_widget(
            'qrc:/tutorials/slice-nav.mp4')
        self.video_players['slice-nav'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # page 5: Shadows
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Shadows")

        page_title = QLabel("Shadow segmentations")
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
        self.video_players['shadow-seg'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
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
        self.video_players['interpolation'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # page 7: Segmentation loading
        page_widget = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_widget.setLayout(page_layout)
        self.tabs.addTab(page_widget, "Load and edit")

        page_title = QLabel("Loading and editing segmentations")
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
        self.video_players['load-seg'] = (vid_layout, vid_widget, vid_player)
        page_widget.layout().addSpacing(20)
        page_widget.layout().addLayout(vid_layout)

        # tutorial navigation buttons
        prev_btn = QPushButton('<')
        next_btn = QPushButton('>')
        prev_btn.clicked.connect(self.prev_page)
        next_btn.clicked.connect(self.next_page)
        nav_btns = QHBoxLayout()
        nav_btns.addWidget(prev_btn)
        nav_btns.addStretch()
        nav_btns.addWidget(next_btn)
        # self.layout().addLayout(nav_btns)

        # button box
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        # TODO: Workaround for a QTBUG-113498
        btn_box.accepted.connect(self.hide)
        self.layout().addWidget(btn_box)

    def closeEvent(self, event: QCloseEvent) -> None:
        # TODO: Workaround for a QTBUG-113498
        self.hide()
        event.ignore()

    def prev_page(self):
        idx = self.tabs.currentIndex() - 1
        if idx < 0:
            idx =  self.tabs.count() - 1
        self.tabs.setCurrentIndex(idx)

    def next_page(self):
        idx = self.tabs.currentIndex() + 1
        if idx >= self.tabs.count():
            idx = 0
        self.tabs.setCurrentIndex(idx)
