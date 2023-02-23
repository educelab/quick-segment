from __future__ import annotations

from math import sqrt
import numpy as np

def calculate_sq_distance(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2

def find_min(point, line):
    min_pt = float('inf')
    for p in line:
        dist = calculate_sq_distance(point, p)
        if dist < min_pt:
            min_pt = dist

    return min_pt

def inverse_vector(vec, orig=[0,0,0]):
    """
    Invert direction of vector based on a reference point
    :param vec: point vector to be inverted
    :param orig: point used as a reference for inverting, if none is given assume vec is a direction
    """
    return [orig[0] + (orig[0] - vec[0]), orig[1] + (orig[1] - vec[1]), vec[2]]

def perpendicular_vector(vec):
    """
    Finds a perpendicular vector to a direction vector
    
    :param vec: vector we want the perpendicular of
    """
    perp = np.cross(vec, [0, 0, 1])
    return [perp[0], perp[1], vec[2]]

def normalized_direction(point, orig = [0,0,0], magnitude = 1):
    """ 
    Return the normalized (magnitude 1) version of a direction vector
    
    :param point: point thta defines vector
    :param orig: point with on to reference the normalization, default to 0,0,0
    :param magnitude: size to be normalized to
    """
    dist = sqrt(calculate_sq_distance(point, orig))

    if dist == 0:
        return [0,0,0]

    x = magnitude * ((point[0] - orig[0]) / dist)
    y = magnitude * ((point[1] - orig[1]) / dist)
    z = magnitude * ((point[2] - orig[2]) / dist)

    return [x,y,z]

def find_sobel_edge(img, point):
    """
    Given an image and a coordinate within it, use edge-detection kernels to return the edge value
    
    :param img: source image, accessed as (y, x)
    :param point: coordinate we want to determine if is edge
    """

    vertical_edge_filter = [[2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2],
                            [4, 2, 0, -2, -4],
                            [2, 1, 0, -1, -2],
                            [2, 1, 0, -1, -2]]

    horizontal_edge_filter = [[-2, -2, -4, -2, -2],
                              [-1, -1, -2, -1, -1],
                              [ 0,  0,  0,  0,  0],
                              [ 1,  1,  2,  1,  1],
                              [ 2,  2,  4,  2,  2]]

    v_edge = 0
    h_edge = 0
    
    for i in range(-2, 2):
        for j in range(-2, 2):
            pixel = img[point[1] + i, point[0] + j]
            v_edge += pixel * vertical_edge_filter[i + 2][j + 2]
            h_edge += pixel * horizontal_edge_filter[i + 2][j + 2]

    return max(v_edge, h_edge)
