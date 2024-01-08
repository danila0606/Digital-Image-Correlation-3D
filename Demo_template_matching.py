# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:56:43 2023

@author: EMSI_CL
"""
# %% Lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pims

from skimage import data
from skimage.feature import match_template
from skimage.io import imread, imshow, imread_collection
import time
from datetime import datetime

# %% Import images
fp = r'.'
fp_img = fp +r'/Crack_PNG_re-ordered/'
frames = pims.ImageSequenceND(fp_img+'*.png', axes_identifiers = ['z', 'w']) # w represents time, because t cause conflict with .tif
frames.bundle_axes = ['z', 'y', 'x']
frames.iter_axes = 'w'
frames

time_size = frames.shape[0]
img_z, img_w, img_h = frames.shape[1], frames.shape[2], frames.shape[3]

# %% function matching
def Match_2_Volume(fp_img, dir0, dir1):
    img_0_dir = fp_img+dir0 # frame 0
    img_1_dir = fp_img+dir1 # frame 1  
    img_0_col = imread_collection(img_0_dir)
    img_1_col = imread_collection(img_1_dir)
    
    tmp_w, tmp_h = 200, 80 
    
    tmp_x_0 = np.arange(100, img_w-tmp_w, 500)
    tmp_y_0 = np.arange(100, img_h-tmp_h, 500)
    tmp_z_0 = np.arange( 5,  img_z,  20)
    
    grid_y_0, grid_z_0, grid_x_0 = np.meshgrid(tmp_y_0,tmp_z_0,tmp_x_0)
    
    grid_0 = np.vstack([grid_x_0.ravel(),grid_y_0.ravel(),grid_z_0.ravel()]).T
    grid_1 = np.full(grid_0.shape, np.nan)
    corr_max = np.full(grid_0.shape[0], np.nan)
    
    for node_id, node_loc in enumerate(grid_0):
        img_0 = imread(img_0_col.files[node_loc[2]])
        tmp_0 = img_0[node_loc[1]:node_loc[1]+tmp_h, node_loc[0]:node_loc[0]+tmp_w]
        
        # loop over near slices
        search_file = img_1_col.files[np.max([node_loc[2]-4,0]):np.min([node_loc[2]+4,img_z])]
        loc_temp = np.zeros([len(search_file),3])
        loc_temp[:,2] = np.arange(np.max([node_loc[2]-4,0]),np.min([node_loc[2]+4,img_z]))
        maxes_temp = np.zeros(len(search_file))
        for img_1_id, img_1_name in enumerate(search_file):
            img_1 = imread(img_1_name)
            res_temp = match_template(img_1, tmp_0)
            # reversed
            loc_temp[img_1_id, 1], loc_temp[img_1_id, 0] = np.unravel_index(np.argmax(res_temp), res_temp.shape)
            maxes_temp[img_1_id] = np.max(res_temp)
            
        # find the maximum in searched z slices    
        corr_max[node_id] = np.max(maxes_temp)
        grid_1[node_id] = loc_temp[np.argmax(maxes_temp)]
                    
    return grid_0, grid_1, corr_max

# %% Loop over stack pairs and save coordinates

for t0 in np.arange(time_size-1):
    t1 = t0 + 1 # Here this "5" is due to the frame jump in the file, which is 5.
    dir0 = r'w' + str(t0) + r'_z*.png'
    dir1 = r'w' + str(t1) + r'_z*.png'
    print(f'Frame {t0:3d} starts at ' + (datetime.now()).strftime("%d/%m/%Y %H:%M:%S"))
    grid_0, grid_1, cc = Match_2_Volume(fp_img, dir0, dir1)
    print(f'Frame {t0:3d} ends   at ' + (datetime.now()).strftime("%d/%m/%Y %H:%M:%S") + '\n')
    np.savetxt(fp + r'/Results/Correlation/Frame' + str(t0) + r'toFrame' + str(t1) + '_grid_0.csv',grid_0, fmt='%.3f', delimiter=',')
    np.savetxt(fp + r'/Results/Correlation/Frame' + str(t0) + r'toFrame' + str(t1) + '_grid_1.csv',grid_1, fmt='%.3f', delimiter=',')
    np.savetxt(fp + r'/Results/Correlation/Frame' + str(t0) + r'toFrame' + str(t1) + '_cc.csv',cc, fmt='%.3f', delimiter=',')

