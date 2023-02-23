#!/usr/bin/env python3
"""Line up"""


def add_arrays(arr1, arr2):
    """Line up"""
    if len(arr1) == len(arr2):
        res=[arr1[x] + arr2[x] for x in range (len(arr1))]
        return res
    else:
        return None
