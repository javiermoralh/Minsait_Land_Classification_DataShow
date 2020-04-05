# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:08:08 2020

@author: javier.moral.hernan1
"""

from math import radians, cos, sin, asin, sqrt


def single_pt_haversine(lat, lng, degrees=True,):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 

    return d