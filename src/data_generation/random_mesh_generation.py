# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:20:53 2022

@author: AlvaroBailletBolivar
"""
import random 
import numpy as np


def randomMeshGen(low_h, high_h):
    '''
    

    Parameters
    ----------
    low_h : int 
        exponent of 2 lower bounding mesh spacing. For example -5 means the lowest spacing in the mesh wil be 2**-5.
    high_h : int
        exponent of 2 lower bounding mesh spacing. For example -3 means the lowest spacing in the mesh wil be 2**-3.

    Returns
    -------
    np.array xgrid. Mesh for problem.

    '''
    low_h = abs(low_h)*-1
    high_h = abs(high_h)*-1
    length = 0
    x = 0
    xgrid = [x]
    while length < 1:
        h = random.uniform(2**low_h, 2**high_h)
        x += h
        xgrid.append(x)
        length += h
        
    xgrid[-1] = 1
    return np.array(xgrid)
