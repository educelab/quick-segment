from __future__ import annotations

import numpy as np
from qs.math import normalize_direction, calculate_sq_distance
from math import sqrt


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

#--------------------------------------------------------------
#               LINEAR INTERPOLATION FUNCTIONS
#--------------------------------------------------------------


def interpolate_point(current, prev_key, next_key):
    """ 
    Returns a new point, interpolad between two points

    :param current: the number of the current slice
    :param prev_key: point in the previous key slice
    :param next_key: point in the next slice
    """
    return [
        find_coordinate(prev_key, next_key, current, 'x'),
        find_coordinate(prev_key, next_key, current, 'y'),
        current
    ]


def verify_interpolation(current, lines):
    """ 
    Verifies if a PARTIAL interpolation is possible between two slices at any given intermediate slice

    :param current: the number of the current slice
    :param lines: an array of lines where each line is a list of points in a key slice
    """
    prev_key = find_previous_key(current, lines)
    next_key = find_next_key(current, lines)

    if prev_key == -1 or next_key == -1:
        return False

    if len(prev_key) != len(next_key):
        return False

    return True


def verify_full_interpolation(lines):
    """ 
    Verifies if a full interpolation is possible between all lines

    :param lines: an array of lines where each line is a list of points in a key slice
    """
    temp = len(lines[list(lines.keys())[0]])
    for line in lines:
        if len(lines[line]) != temp:
            return False
        temp = len(lines[line])
    return True

def full_interpolation(lines):
    """ 
    Interpolates the full extent of the segmentation

    :param lines: an array of lines where each line is a list of points in a key slice
    """

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



#--------------------------------------------------------------
#         NON LINEAR INTERPOLATION FUNCTIONS (for now helpers)
#--------------------------------------------------------------

def find_normal_direction(point, n1, n2):
    """ 
    Finds and returns the normal of the point based on its neighbors

    :param point: a point which you want the normal of
    :param n1: the neighboring point
    :param n2: the other neighboring point
    """

    # Normalize the the direction vectors that point to the neighbors
    n1 = normalize_direction(n1, point, 20)
    n2 = normalize_direction(n2, point, 20)

    pivot = [point[0] + (n1[0] + n2[0]),
                 point[1] + (n1[1] + n2[1]),
                 point[2] + (n1[2] + n2[2])]

    pivot = normalize_direction(pivot, point, 30)

    return pivot

    

