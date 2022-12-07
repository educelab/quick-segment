from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


def fill_seg_list(self, vol, paths_dir, lst):
    for seg in os.listdir(paths_dir):
        f = os.path.join(paths_dir, seg)
        # checking if it is a file
        if os.path.isdir(f) and os.path.isfile(
                os.path.join(f, 'pointset.vcps')) and (
                seg != 'fromInterpolator'):
            item = QtWidgets.QListWidgetItem(seg, lst).setCheckState(
                Qt.CheckState.Unchecked)  # can not add listeners to the QListItems but can check their state


def load_vcps(dir_path, seg):
    with open(os.path.join(dir_path, seg) + "/pointset.vcps", 'rb') as file:
        # print(np.memmap(file, dtype=np.double, mode='r', offset=7))
        points = file.read().split(b'<>\n')[1]
        return cloud_to_dict(np.frombuffer(points, dtype='float64'))


def cloud_to_dict(cloud):
    cloud = np.reshape(cloud, (-1, 3))
    lines = dict()

    for point in cloud:
        slice = int(point[2])
        point = [point[0], point[1], slice]
        lines.setdefault(int(slice), []).append(point)

    # initialize direction vector
    x_diff = 0
    y_diff = 0

    first_key = list(lines.keys())[0]
    last_key = list(lines.keys())[-1]
    for line in range(first_key + 1, last_key + 1):
        # If there is a change in direction that means it is a key slice
        if (round(lines[line][0][0] - lines[line - 1][0][0], 9) == x_diff) and (
                round(lines[line][0][1] - lines[line - 1][0][1], 9) == y_diff):
            x_diff = round(lines[line][0][0] - lines[line - 1][0][0], 9)
            y_diff = round(lines[line][0][1] - lines[line - 1][0][1], 9)
            del lines[line - 1]
        else:
            x_diff = round(lines[line][0][0] - lines[line - 1][0][0], 9)
            y_diff = round(lines[line][0][1] - lines[line - 1][0][1], 9)

    # # Printing to verify points
    # for line in lines:
    #     print(line, ": ", lines[line])

    return lines


def load_json(dir, seg):
    with Path(os.path.join(dir, seg) + "/pointset.json").open('r') as file:
        # print(np.memmap(file, dtype=np.double, mode='r', offset=7))
        data = json.load(file, object_hook=lambda d: {int(k): v for k, v in
                                                      d.items()})

    return data


def get_date():
    t = datetime.now()
    return f'{t.year}{t.month}{t.day}{t.hour}{t.minute}{t.second}'


def get_segmentation_dir(paths_dir, uuid):
    dirname = uuid
    path = os.path.join(paths_dir, dirname)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def write_ordered_vcps(path, pointset):
    # Open output file and write ASCII header
    file_path = Path(path) / "pointset.vcps"
    with file_path.open('wt') as file:
        file.writelines([
            f'width: {pointset.shape[1]}\n',
            f'height: {pointset.shape[0]}\n',
            f'dim: {pointset.shape[2]}\n',
            'ordered: true\n',
            'type: double\n',
            'version: 1\n',
            '<>\n'
        ])

    # Reopen in binary append mode
    with file_path.open('ab') as file:
        # Write as doubles
        pointset.tofile(file)


def write_seg_json(path, pointset):
    # Open output file and write ASCII header
    with open(path + "/pointset.json", 'w', encoding='utf-8') as file:
        json.dump(pointset, file)


def write_metadata(path, uuid):
    # TODO: Hardcoded volume path
    vol = (os.listdir(path + "/../../volumes")[1])
    data = {
        "name": uuid,
        "type": "seg",
        "uuid": uuid,
        "vcps": "pointset.vcps",
        "volume": vol
    }

    with open(path + "/meta.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))
