from __future__ import annotations

from math import sqrt
import numpy as np
import cv2 as cv

def calculate_sq_distance(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2

def find_min(point, line):
    min_pt = float('inf')
    for p in line:
        dist = calculate_sq_distance(point, p)
        if dist < min_pt:
            min_pt = dist

    return min_pt

def get_vector_magnitude(vec, orig=[0,0,0]):
    """
    Get magnitude of vector based on a reference point

    :param vec: point vector
    :param orig: point used as a reference, if none is given assume vec is a direction
    """
    return sqrt(calculate_sq_distance(vec, orig))


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

def sobel_edge_detection_img(image):
    """
    Finds Sobel edges for all pixels in image
    
    :param img: image to be used
    """
    new_img = np.full_like(image, 1)
    for row in range(3, image.shape[0] - 3):
        for pixel in range(3, image.shape[1] - 3):
            new_img[row][pixel] = find_sobel_edge(image, [pixel, row])
    
    return new_img

def canny_edge(image, t1=100, t2=120):
    """
    Wrapper around OpenCV's canny edge detection to convert nparray to the 
    right format and normalize it between 0 and 255 before sending to the
    canny edge detection

    :param img: image to be used
    :param t1: threshold 1 for the edge detection
    :param t2: threhold 2 for the edge detection
    """
    img = image.astype('float64')
    img *= (255.0/img.max())
    img = np.uint8(img)

    return cv.Canny(image=img, threshold1=t1, threshold2=t2)