from __future__ import annotations

from math import sqrt

def calculate_sq_distance(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2


def find_min(point, line):
    min_pt = float('inf')
    for p in line:
        dist = calculate_sq_distance(point, p)
        if dist < min_pt:
            min_pt = dist

    return min_pt

def normalize_direction(point, orig = [0,0,0], magnitude = 1):
    """ 
    Return the normalized (magnitude 1) version of a direction vector
    
    :param point: point thta defines vector
    :param orig: point with on to reference the normalization, default to 0,0,0
    :param magnitude: size to be normalized to
    """

    dist = sqrt(calculate_sq_distance(point, orig))

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
