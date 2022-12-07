from __future__ import annotations

import numpy as np


def find_coordinate(start, end, current, coord):
    i = 1 if coord == 'y' else 0
    return (end[i] - start[i]) * (current - start[2]) / (end[2] - start[2]) + \
           start[i]


def find_next_key(current, lines):
    temp = lines.copy()
    temp[current] = 1
    slices = sorted(temp.keys())

    pos = slices.index(current)
    if pos >= (len(slices) - 1):
        return -1

    return temp[slices[pos + 1]]


def find_previous_key(current, lines):
    temp = lines.copy()
    temp[current] = 1
    slices = sorted(temp.keys())

    pos = slices.index(current)
    if pos <= 0:
        return -1

    return temp[slices[pos - 1]]


def interpolate_point(current, prev_key, next_key):
    return [
        find_coordinate(prev_key, next_key, current, 'x'),
        find_coordinate(prev_key, next_key, current, 'y'),
        current
    ]


def verify_interpolation(current, lines):
    prev_key = find_previous_key(current, lines)
    next_key = find_next_key(current, lines)

    if prev_key == -1 or next_key == -1:
        return False

    if len(prev_key) != len(next_key):
        return False

    return True


def verify_full_interpolation(lines):
    temp = len(lines[list(lines.keys())[0]])
    for line in lines:
        if len(lines[line]) != temp:
            return False
        temp = len(lines[line])
    return True


def full_interpolation(lines):
    if len(lines) <= 1:
        print('add points to at least two separate slices')
        return

    lines = dict(sorted(lines.items()))

    first = list(lines.keys())[0]
    last = list(lines.keys())[-1]

    # empty point cloud to store final points
    cloud = [[]]

    # Set starting parameters
    prev_key = lines[first]
    next_key = find_next_key(first, lines)
    for i, slice_idx in enumerate(range(first + 1, last)):
        cloud.append([])

        # When we reach the 'next' slice (a user defined slice), add it and update boundaries
        if slice_idx == next_key[0][2]:
            for point in lines[slice_idx]:
                cloud[i].append(point)
            next_key = find_next_key(slice_idx, lines)
            prev_key = cloud[i]
        else:
            # If slice is not user defined, find it through interpolation
            for point in range(len(prev_key)):
                interpolated_point = interpolate_point(slice_idx,
                                                       prev_key[point],
                                                       next_key[point])
                cloud[i].append(interpolated_point)

    # Conclude interpolation by adding last slice
    for point in lines[last]:
        cloud[-1].append(point)

    return np.array(cloud, dtype='float64')
