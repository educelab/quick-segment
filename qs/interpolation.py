from __future__ import annotations

import numpy as np
from qs.math import (normalized_direction,
                    calculate_sq_distance,
                    get_vector_magnitude,
                    find_sobel_edge, inverse_vector, 
                    perpendicular_vector, 
                    canny_edge
                    )
from math import sqrt
from matplotlib import pyplot as plt

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
    Returns a new point, linearly interpolated between two points

    :param current: the number of the current slice
    :param prev_key: point in the previous key slice
    :param next_key: point in the next slice
    """
    return [
        find_coordinate(prev_key, next_key, current, 'x'),
        find_coordinate(prev_key, next_key, current, 'y'),
        current
    ]

def partial_linear_interpolation(ax, lines, slice, img, circle_size=0):
    """
    Partially linearly interpolates a given slice between the two slices that
    surround it.

    :param ax: the ax on which to draw interpolation
    :param lines: the segmentation lines
    :param slice: slice where the interpolation is drawn
    :param circle_size: the circle that indicates if the line is active (0 if not, 7 if active)
    """
    previous_key = find_previous_key(slice, lines)
    next_key = find_next_key(slice, lines)

    point = interpolate_point(slice, previous_key[0], next_key[0])
    prev_point = point # Initialize it to the point, update later
    for i in range(0, len(previous_key) - 1):
        next_point = interpolate_point(slice, previous_key[i + 1], next_key[i + 1])
        ax.plot([point[0], next_point[0]],
                        [point[1], next_point[1]], color='yellow')
        ax.add_artist(
            plt.Circle((point[0], point[1]), 3.5, color='yellow'))
        ax.add_artist(
            plt.Circle((point[0], point[1]), circle_size,
                        facecolor='none', edgecolor='yellow'))

        prev_point = point
        point = next_point

    last_point = interpolate_point(slice, previous_key[-1],
                                    next_key[-1])
    ax.add_artist(
        plt.Circle((last_point[0], last_point[1]), 3.5,
                    color='yellow'))
    ax.add_artist(
        plt.Circle((last_point[0], last_point[1]), circle_size,
                    facecolor='none', edgecolor='yellow'))

def full_linear_interpolation(lines):
    """ 
    Linearly interpolates the full extent of the segmentation

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

        # When we reach the 'next' key slice (a user defined non-interpolated slice), add it and update boundaries
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
#          NON LINEAR INTERPOLATION FUNCTIONS
#--------------------------------------------------------------

def find_normal_direction(point, n1, n2):
    """ 
    Finds and returns the normal of the point based on its neighbors

    :param point: a point which you want the normal of
    :param n1: the neighboring point
    :param n2: the other neighboring point, if none is given assign it -1, to be reassigned later
    """
    # Normalize the the direction vectors that point to the neighbors
    n1 = normalized_direction(n1, point, 20)
    n2 = normalized_direction(n2, point, 20)

    direction = [(n1[0] + n2[0]),
                 (n1[1] + n2[1]),
                 (n1[2] + n2[2])]

    if (direction == [0,0,0]):
        direction = perpendicular_vector(n1)

    return normalized_direction(direction, magnitude=40)


def detect_edge_along_line(edge_data, point, direction, magnitude=40):
    vec_size = get_vector_magnitude(direction)
    scaled_direction = [(direction[0])/vec_size, (direction[1])/vec_size, 0]
    x = point[0]
    y = point[1]

    end_x = point[0] + direction[0]
    end_y = point[1] + direction[1]

    for i in range(magnitude):
        x = x + scaled_direction[0]
        y = y + scaled_direction[1]
        if (edge_data[int(y)][int(x)] >= 10):
            return [x, y, point[2]]

    return -1 # edge not found, return -1

def adjust_point_based_on_edges(edge_data, point, neighbor_1, neighbor_2=None, magnitude=40):
    if (neighbor_2 == None):
        neighbor_2 = inverse_vector(neighbor_1, point)

    normal_direction = find_normal_direction(point, neighbor_1, neighbor_2)
    edge_1 = detect_edge_along_line(edge_data, point, normal_direction, magnitude=magnitude)
    edge_2 = detect_edge_along_line(edge_data, point, inverse_vector(normal_direction),  magnitude=magnitude)

    if (edge_1 == -1 or edge_2 == -1): # Either of the edges are not found, if this is the case, do not adjust point
        midpoint = point
    else:
        midpoint = [(edge_1[0] + edge_2[0])/2, (edge_1[1] + edge_2[1])/2, point[2]]

    return midpoint

def draw_detected_edge(ax, edge_data, point, neighbor_1, neighbor_2=None, magnitude=40):
    if (neighbor_2 == None):
        neighbor_2 = inverse_vector(neighbor_1, point)

    normal_direction = find_normal_direction(point, neighbor_1, neighbor_2)
    edge_1 = detect_edge_along_line(edge_data, point, normal_direction, magnitude=magnitude)
    edge_2 = detect_edge_along_line(edge_data, point, inverse_vector(normal_direction), magnitude=magnitude)

    if (edge_1 != -1 and edge_2 != -1): 
        ax.add_artist(
            plt.Circle((edge_1[0], edge_1[1]), 2,
                    facecolor='magenta'))
        ax.add_artist(
            plt.Circle((edge_2[0], edge_2[1]), 2,
                    facecolor='magenta'))
        ax.plot([point[0],edge_1[0]],
                [point[1], edge_1[1]], color='magenta')
        ax.plot([point[0], edge_2[0]],
                    [point[1], edge_2[1]], color='magenta')

def partial_nonlinear_interpolation(ax, lines, slice, vol, draw_edges=True, edge_threshold1=100, edge_threshold2=120, edge_search_limit=40, circle_size=0):
    """
    Partially interpolates a given slice between the two slices that
    surround it based on edges.

    :param ax: the ax on which to draw interpolation
    :param lines: the segmentation lines
    :param slice: slice where the interpolation is drawn
    :param vol: images of the slices used to calculate edges
    :param circle_size: the circle that indicates if the line is active (0 if not, 7 if active)
    """
    previous_key = find_previous_key(slice, lines)
    next_key = find_next_key(slice, lines)
    relative_key = [i for i in previous_key]
    next_relative_key = []

    initial_slice = previous_key[0][2] + 1

    for i in range(initial_slice, slice):
        next_relative_key.clear()
        # For each intermediate slice between the previous and current
        # Get edge information
        edge_data = canny_edge(vol[slice], edge_threshold1, edge_threshold2, dilation=2)
        # Find first two points
        point = interpolate_point(i, relative_key[0], next_key[0])
        next_point = interpolate_point(i, relative_key[1], next_key[1])

        # Adjust first point
        next_relative_key.append(adjust_point_based_on_edges(edge_data, point=point, neighbor_1=next_point, magnitude=edge_search_limit))

        # Iterate between the 2nd and penultimate point
        for j in range(1, len(relative_key) - 1):
            prev_point = point
            point = next_point
            next_point = interpolate_point(i, relative_key[j + 1], next_key[j + 1])

            next_relative_key.append(adjust_point_based_on_edges(edge_data, point=point, neighbor_1=prev_point, neighbor_2=next_point, magnitude=edge_search_limit))

        # Adjust last point
        next_relative_key.append(adjust_point_based_on_edges(edge_data, point=next_point, neighbor_1=point, magnitude=edge_search_limit))
        relative_key.clear()
        relative_key = [p for p in next_relative_key]
    
    # For last slice, repeat above process on last time, but this time draw it to canvas
    edge_data = canny_edge(vol[slice], edge_threshold1, edge_threshold2, dilation=2)
    point = interpolate_point(slice, relative_key[0], next_key[0])
    next_point = interpolate_point(slice, relative_key[1], next_key[1])
    adjusted_point = adjust_point_based_on_edges(edge_data, point=point, neighbor_1=next_point, magnitude=edge_search_limit)
    if draw_edges: draw_detected_edge(ax, edge_data, point=point, neighbor_1=next_point, magnitude=edge_search_limit)
    ax.add_artist(
        plt.Circle((adjusted_point[0], adjusted_point[1]), 3.5, color='yellow'))

    for j in range(1, len(relative_key) - 1):
        prev_point = point
        point = next_point
        next_point = interpolate_point(slice, relative_key[j + 1], next_key[j + 1])
        adjusted_point = adjust_point_based_on_edges(edge_data, point=point, neighbor_1=prev_point, neighbor_2=next_point, magnitude=edge_search_limit)
        if draw_edges: draw_detected_edge(ax, edge_data, point=point, neighbor_1=prev_point, neighbor_2=next_point, magnitude=edge_search_limit)
        ax.add_artist(
            plt.Circle((adjusted_point[0], adjusted_point[1]), 3.5, color='yellow'))
    
    adjusted_point = adjust_point_based_on_edges(edge_data, point=next_point, neighbor_1=point, magnitude=edge_search_limit)
    if draw_edges: draw_detected_edge(ax, edge_data, point=next_point, neighbor_1=point, magnitude=edge_search_limit)
    ax.add_artist(
        plt.Circle((adjusted_point[0], adjusted_point[1]), 3.5, color='yellow'))

def full_nonlinear_interpolation(lines, vol, edge_threshold1=100, edge_threshold2=120, edge_search_limit=40):
    """
    Fully interpolates a given slice between the two slices that
    surround it based on edges.

    :param lines: the segmentation lines
    :param vol: images of the slices used to calculate edges
    """

    if len(lines) <= 1:
        print('add points to at least two separate slices')
        return
    
    lines = dict(sorted(lines.items()))

    first = list(lines.keys())[0]
    last = list(lines.keys())[-1]

    # empty point cloud to store final points
    cloud = []

    prev_key = lines[first]
    next_key = find_next_key(first, lines)
    relative_key = [i for i in prev_key]
    next_relative_key = []

    for i, slice_idx in enumerate(range(first, last)):

        if slice_idx == next_key[0][2]:
            for point in lines[slice_idx]:
                cloud[i].append(point)
            next_key = find_next_key(slice_idx, lines)
            relative_key = cloud[i]
        else:
            next_relative_key.clear()
            # For each intermediate slice between the previous and current
            # Get edge information
            edge_data = canny_edge(vol[slice_idx], edge_threshold1, edge_threshold2, dilation=2)
            # Find first two points
            point = interpolate_point(i, relative_key[0], next_key[0])
            next_point = interpolate_point(i, relative_key[1], next_key[1])

            # Adjust first point
            next_relative_key.append(adjust_point_based_on_edges(edge_data, point=point, neighbor_1=next_point, magnitude=edge_search_limit))

            # Iterate between the 2nd and penultimate point
            for j in range(1, len(relative_key) - 1):
                prev_point = point
                point = next_point
                next_point = interpolate_point(i, relative_key[j + 1], next_key[j + 1])

                next_relative_key.append(adjust_point_based_on_edges(edge_data, point=point, neighbor_1=prev_point, neighbor_2=next_point, magnitude=edge_search_limit))

            # Adjust last point
            next_relative_key.append(adjust_point_based_on_edges(edge_data, point=next_point, neighbor_1=point, magnitude=edge_search_limit))
            relative_key.clear()
            relative_key = [p for p in next_relative_key]
            cloud.append([p for p in next_relative_key])

        print(next_relative_key)
        print(cloud[-1])

    # Conclude interpolation by adding last slice
    cloud.append([])
    for point in lines[last]:
        cloud[-1].append(point)

    return np.array(cloud, dtype='float64')


#--------------------------------------------------------------
#              GENERAL INTERPOLATION FUNCTIONS
#--------------------------------------------------------------

# VERIFY INTERPOLATION FUCNTIONS ------------------------------
def verify_partial_interpolation(current, lines):
    """ 
    Verifies if a partial interpolation is possible between two slices at a given intermediate slice

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
    Verifies if a full interpolation is possible across all lines

    :param lines: an array of lines where each line is a list of points in a key slice
    """
    temp = len(lines[list(lines.keys())[0]])
    for line in lines:
        if len(lines[line]) != temp:
            return False
        temp = len(lines[line])
    return True

# INTERPOLATION FUNCTIONS -------------------------------------
def partial_interpolation(ax, lines, slice, type='linear', vol=None, draw_edges=True, edge_threshold1=100, edge_threshold2=120, edge_search_limit=40, circle_size=0):
    """
    Interpolates all points in a line between two slices
    
    :param ax: the ax on which to draw interpolation
    :param lines: the segmentation lines
    :param type: the type of interpolation to be carried out (linear or non-linear)
    :param vol: images of the slices used to calculate edges
    :param draw_edges: where or not to draw the normals the make up the edge detection
    :param edge_threshold1: lower threshold for the canny edge detection
    :param edge_threshold2: higher threshold for the canny edge detection
    :param edge_search_limit: maximum distance away from point to look for edge
    :param circle_size: the circle that indicates if the line is active (0 if not, 7 if active)
    """

    if type == 'linear':
        partial_linear_interpolation(ax, lines, slice, circle_size)
    elif type == 'non-linear' and vol != None:
        partial_nonlinear_interpolation(ax, lines, slice, vol, 
                                        draw_edges=draw_edges, 
                                        edge_threshold1=edge_threshold1, 
                                        edge_threshold2=edge_threshold2, 
                                        edge_search_limit=edge_search_limit, 
                                        circle_size=circle_size)
    else:
        print("Not accepted interpolation type")

def full_interpolation(lines, type='linear', vol=None, edge_threshold1=100, edge_threshold2=120, edge_search_limit=40):
    """
    Interpolates all points in a segmentation
    
    :param lines: the segmentation lines
    :param type: the type of interpolation to be carried out (linear or non-linear)
    :param vol: images of the slices used to calculate edges
    :param edge_threshold1: lower threshold for the canny edge detection
    :param edge_threshold2: higher threshold for the canny edge detection
    :param edge_search_limit: maximum distance away from point to look for edge
    """

    if type == 'linear':
        return full_linear_interpolation(lines)
    elif type == 'non-linear' and vol != None:
        return full_nonlinear_interpolation(lines, vol, edge_threshold1=edge_threshold1, edge_threshold2=edge_threshold2, edge_search_limit=edge_search_limit)
    else:
        print("Not accepted interpolation type")