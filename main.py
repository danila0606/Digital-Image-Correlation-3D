import muDIC as dic

import numpy as np
import matplotlib.pyplot as plt

import config 
from config import Config

import time
import os

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters

cfg = Config()

image_stacks_t = []
image_z_max = int(1e10)

for path in cfg.assets:
    image_stack = os.listdir(path)
    image_stack.sort()
    image_z_max = min(image_z_max, len(image_stack))
    image_stacks_t.append(image_stack)

if cfg.subset_offset < cfg.subset_size:
    raise TypeError("Offset between subsets must be bigger or equal than subset size!")


subset_number_x = int((cfg.region_of_interest_xy_max[1] - cfg.region_of_interest_xy_min[1] - cfg.subset_size) // (cfg.subset_offset)) + 1
subset_number_y = int((cfg.region_of_interest_xy_max[0] - cfg.region_of_interest_xy_min[0] - cfg.subset_size) // (cfg.subset_offset)) + 1
print("Subset number_x: ", subset_number_x, ", number_y: ", subset_number_y)


reference_time = 0
initital_z_guess = 0
analyze_result = np.zeros((cfg.sub_group_size * subset_number_y, cfg.sub_group_size * subset_number_x, len(range(0, image_z_max, cfg.z_bounce)), 3))

# Searching Crack in the image
img_path = cfg.assets[cfg.search_crack_asset_id] + cfg.search_crack_img_name
im = (imread(img_path))
grads = filters.sobel(im)
crack_grad_threshold = 0.5 * grads.mean()

def is_subset_in_crack(xy_min, grads, crack_grad_threshold) :
    subset_grads = grads[xy_min[0] : xy_min[0] + cfg.subset_size, xy_min[1] : xy_min[1] + cfg.subset_size]
    if (subset_grads.mean() < crack_grad_threshold) :
        return True
    
    return False


start_time = time.time()

mesher = dic.MyMesher(type='q4')
for i in range (0, subset_number_y) :
    for j in range (0, subset_number_x):
        subset_center = cfg.region_of_interest_xy_min + (cfg.subset_size // 2 + cfg.subset_offset * np.array([i, j]))
        s_xy_min = subset_center - cfg.subset_size / 2
        s_xy_max = subset_center + cfg.subset_size / 2

        if (is_subset_in_crack(s_xy_min.astype(int), grads, crack_grad_threshold)) :
            analyze_result[cfg.sub_group_size*i : cfg.sub_group_size*(i+1), cfg.sub_group_size*j : cfg.sub_group_size*(j+1), :] = np.array([None, None, None])
            continue


        for k in range(0, image_z_max, cfg.z_bounce) :
            init_z = k + initital_z_guess
            if (init_z < 0) :
                init_z = 0
            elif (init_z >= image_z_max) :
                init_z = image_z_max - 1

            search_z_min = init_z - cfg.z_radius
            if (search_z_min < 0) :
                search_z_min = 0
            
            search_z_max = init_z + cfg.z_radius
            if (search_z_max >= image_z_max) :
                search_z_max = image_z_max - 1

            best_z = search_z_min
            cur_koef = 1e5
            best_results = None

            image_k = plt.imread(cfg.assets[reference_time] + image_stacks_t[reference_time][k])
            mesh = mesher.my_mesh(dic.image_stack_from_list([image_k]), Xc1=s_xy_min[1], Xc2=s_xy_max[1], Yc1=s_xy_min[0], Yc2=s_xy_max[0], n_elx=cfg.sub_group_size, n_ely=cfg.sub_group_size)

            for z in range (search_z_min, search_z_max + 1) :             
                img_stack_tmp = dic.image_stack_from_list([image_k, plt.imread(cfg.assets[reference_time + 1] + image_stacks_t[reference_time + 1][z])])

                inputs = dic.DICInput(mesh,img_stack_tmp, maxit=cfg.maxit)
                dic_job = dic.MyDICAnalysis(inputs)

                results, koefs = dic_job.run()

                deformed_id = cfg.image_t_max - 1

                if (koefs[deformed_id] < cur_koef) :
                    cur_koef = koefs[deformed_id]
                    best_z = z
                    best_results = results

            fields = dic.Fields(best_results)
            disps = fields.disp()
            u = disps[0, 0, :, :, deformed_id]
            v = disps[0, 1, :, :, deformed_id]

            uvz = np.array([u.flatten(), v.flatten(), np.full(cfg.sub_group_size ** 2, best_z - k)]).T.reshape(cfg.sub_group_size, cfg.sub_group_size, 3)

            analyze_result[cfg.sub_group_size*i : cfg.sub_group_size*(i+1), cfg.sub_group_size*j : cfg.sub_group_size*(j+1), k // cfg.z_bounce] = uvz

            #TODO :
            # initital_z_guess = ...
            #
            #         

bounced_z_img_id = cfg.show_img_id // cfg.z_bounce

xs = np.arange(cfg.region_of_interest_xy_min[1], 1 + cfg.region_of_interest_xy_max[1] - cfg.subset_size, cfg.subset_offset) + cfg.subset_size // 2
ys = np.arange(cfg.region_of_interest_xy_min[0], 1 + cfg.region_of_interest_xy_max[0] - cfg.subset_size, cfg.subset_offset) + cfg.subset_size // 2
plot_z = analyze_result[:, :, bounced_z_img_id, 2]

print("PLOT Z:", plot_z)
print('TIME SPENT:', time.time() - start_time)
print("xs:", xs, "ys", ys)


background_image = plt.imread(cfg.assets[0] + image_stacks_t[0][cfg.show_img_id])
plt.imshow(background_image, cmap=plt.cm.gray, origin="lower", extent=(0, cfg.image_x_max, 0, cfg.image_y_max))
plt.pcolormesh(xs, ys, plot_z, alpha=0.3)
plt.colorbar()
plt.show()
