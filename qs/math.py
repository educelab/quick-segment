from __future__ import annotations


def calculate_sq_distance(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2


def find_min(point, line):
    min_pt = float('inf')
    for p in line:
        dist = calculate_sq_distance(point, p)
        if dist < min_pt:
            min_pt = dist

    return min_pt
