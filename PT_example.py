import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('figure',  figsize=(10, 6))
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"

import os
import numpy as np
import pandas as pd
import pickle
import pims
import trackpy as tp
from Linear_Nearest_Interp import LinearNDInterpolatorExt

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# Calibration
scale_x = 0.43 # um / pixel
scale_y = scale_x
scale_z = 5    # um / pixel
px_to_um = np.array([scale_x, scale_y, scale_z])

def closest_node(node, nodes, m): # Nearest m particels
    nodes = np.array(nodes)
    dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
    return np.argsort(dist_2)[1:m+1]

def RemoveWrongMatch(grid_1, grid_0,corr_max, k, q1, q3, corr_th, max_ux):
    grid_0 = grid_0[~np.isnan(corr_max),:]
    grid_1 = grid_1[~np.isnan(corr_max),:]
    corr_max = corr_max[~np.isnan(corr_max)]
    disp = grid_1 - grid_0
    
    mask = np.full(len(grid_0),True)
    mask[corr_max<corr_th] = False
    mask[np.abs(disp[:,0])>max_ux] = False
    for node_id, node in enumerate(grid_0):
        neighbors = closest_node(node, grid_0, k) # compare the nearest n neighbors
        Q1 = np.quantile(disp[neighbors,0], q1)
        Q3 = np.quantile(disp[neighbors,0], q3)
        IQR = Q3-Q1
        if disp[node_id,0]>Q3+1.5*IQR or disp[node_id,0]<Q1-1.5*IQR:
            mask[node_id]=False
    
    grid_0 = grid_0[mask]
    grid_1 = grid_1[mask]
    corr_max = corr_max[mask]
    return grid_0, grid_1, corr_max

fp = r'.'
fp_img = fp+r'/Crack_PNG_re-ordered_Test/'

frames = pims.ImageSequenceND(fp_img+'*.png', axes_identifiers = ['z', 'w']) # w represents time, because t cause conflict with .tif
frames.bundle_axes = ['z', 'y', 'x']
frames.iter_axes = 'w'

time_size = frames.shape[0]

p_diameter = [19, 19, 27]
p_separation = [10, 10, 15]
# Particle locations path
pt_loc_path = fp+r'/Results/Locating_Test/pt_loc_' + ''.join(str(p_dim) for p_dim in p_diameter) + r'_' + ''.join(str(p_sep) for p_sep in p_separation) + r'.pkl'

if not os.path.exists(pt_loc_path) :   
    pt_loc = tp.batch(frames, diameter=p_diameter, separation=p_separation)
    pt_loc.to_pickle(pt_loc_path)

pt_loc = pd.read_pickle(pt_loc_path)

# Calibrate
pt_loc['xum'] = pt_loc['x'] * scale_x
pt_loc['yum'] = pt_loc['y'] * scale_y
pt_loc['zum'] = pt_loc['z'] * scale_z

pt_search_range = 30
disp_interps = []
disp_source_method = 'TM' # TM

if (disp_source_method == 'TM') :

    for tt0 in np.arange(time_size-1):
        tt1 = tt0 + 1
        grid_0 = np.loadtxt(fp+r'/Results/Correlation/Frame'+str(tt0)+'toFrame'+str(tt1)+'_grid_0.csv', delimiter=',')
        grid_1 = np.loadtxt(fp+r'/Results/Correlation/Frame'+str(tt0)+'toFrame'+str(tt1)+'_grid_1.csv', delimiter=',')
        corr   = np.loadtxt(fp + r'/Results/Correlation/Frame'+str(tt0)+'toFrame'+str(tt1)+'_cc.csv', delimiter=',')
        grid_0, grid_1, corr = RemoveWrongMatch(grid_1, grid_0,corr, 10, 0.25, 0.75, corr_th=0.3, max_ux=300)
        grid_0 = grid_0 * px_to_um
        grid_1 = grid_1 * px_to_um
        disp_01 = grid_1 - grid_0
        disp_interps.append(LinearNDInterpolatorExt(grid_0, disp_01))

        pt_link_path = fp+r'/Results/Linking_Test/pt_link_r' + str(pt_search_range) + r'um_corr.pkl'
elif (disp_source_method == 'DIC') :
    for tt0 in np.arange(time_size-1):
        tt1 = tt0 + 1
        grid_0 = np.load(fp+r'/output/all_disps_200_no_downsampling/ref_'+str(tt0)+r'_'+str(tt1)+'.npy')
        grid_1 = np.load(fp+r'/output/all_disps_200_no_downsampling/def_'+str(tt0)+r'_'+str(tt1)+'.npy')
        corr   = np.load(fp+r'/output/all_disps_200_no_downsampling/corr_'+str(tt0)+r'_'+str(tt1)+'.npy')
        # grid_0, grid_1, corr = RemoveWrongMatch(grid_1, grid_0,corr, 10, 0.25, 0.75, corr_th=0.3, max_ux=300)
        grid_0 = grid_0 * px_to_um
        grid_1 = grid_1 * px_to_um
        disp_01 = grid_1 - grid_0
        disp_interps.append(LinearNDInterpolatorExt(grid_0, disp_01))

        pt_link_path = fp+r'/Results/DIC_Linking_Test/pt_link_r' + str(pt_search_range) + r'um_corr.pkl'
else :
    raise ValueError(disp_source_method + ' doesn\'t supported!')

if not os.path.exists(pt_link_path) :
    @tp.predict.predictor
    def pred01(t1, particle):
        pos_pred_t1 = particle.pos + disp_interps[particle.t](particle.pos)[0]
        return pos_pred_t1
    
    # For data structure
    fTuple = ()
    for fid in np.arange(np.max(pt_loc.frame)+1):
        fTuple = fTuple + (pt_loc[pt_loc['frame']==fid],)
    # Link particles
    pt_link = pd.concat(tp.link_df_iter(fTuple, search_range = pt_search_range, pos_columns = ['xum', 'yum', 'zum'],
                                     adaptive_stop = 0.01, adaptive_step = 0.95, predictor=pred01))
    
    # Save linked
    pt_link.to_pickle(pt_link_path)


linked3D = pd.read_pickle(pt_link_path)

ref_fn = 0
ev_fn = 3

F0 = linked3D[linked3D['frame']==ref_fn].set_index('particle')
F1 = linked3D[linked3D['frame']==ev_fn].set_index('particle')

F0 = F0[F0.index.isin(F1.index)].sort_index()
F1 = F1[F1.index.isin(F0.index)].sort_index()

X0_img = F0[['x', 'y', 'z']].to_numpy()
x1_img = F1[['x', 'y', 'z']].to_numpy()

X0_um = X0_img * px_to_um
x1_um = x1_img * px_to_um
disp = x1_um - X0_um

ev_slice = 60

pt_filter = np.logical_and(X0_img[:,2]>40,X0_img[:,2]<50,)

fig0, ax0 = plt.subplots(figsize=(6,3),dpi=300)
ax0.imshow(frames[ev_fn][ev_slice],cmap='gray')
# sc0 = ax0.scatter(x1_img[pt_filter,0],x1_img[pt_filter,1], c=disp[pt_filter,1]-np.nanmean(disp[pt_filter,1]),cmap='jet',s=1)
sc0 = ax0.scatter(x1_img[:,0],x1_img[:,1], c=disp[:,1]-np.nanmean(disp[:,1]),cmap='jet',s=1)

sc0.set_clim(-50,50)
cbar0 = fig0.colorbar(sc0,ax=ax0)
ax0.set_aspect('equal', adjustable='box')
ax0.axis('off')

plt.show()

