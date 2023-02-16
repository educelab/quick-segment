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

def normalize_direction(point, orig = [0,0,0], size = 1):
    """ 
    Return the normalized (magnitude 1) version of a direction vector
    
    :param point: point thta defines vector
    :param orig: point with on to reference the normalization, default to 0,0,0
    :param size: size to be normalized to
    """

    dist = sqrt(calculate_sq_distance(point, orig))

    x = size * ((point[0] - orig[0]) / dist)
    y = size * ((point[1] - orig[1]) / dist)
    z = size * ((point[2] - orig[2]) / dist)

    return [x,y,z]