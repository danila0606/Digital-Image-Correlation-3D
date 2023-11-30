import muDIC as dic

import numpy as np
import matplotlib.pyplot as plt

import config 
from config import Config


from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters

cfg = Config()

image_stacks_t = []
image_z_max = int(1e10)

for path in cfg.assets:
    image_stack = dic.image_stack_from_folder(path, file_type=cfg.image_type)
    image_z_max = min(image_z_max, image_stack.__len__())
    image_stacks_t.append(image_stack)

image_stacks_z = []

for i in range(0, image_z_max) :
    image_stack_z = [image_stacks_t[k][i] for k in range (0, cfg.image_t_max)]
    image_stacks_z.append(dic.image_stack_from_list(image_stack_z))


if cfg.subset_offset < cfg.subset_size:
    raise TypeError("Offset between subsets must be bigger or equal than subset size!")


subset_number_x = int((cfg.region_of_interest_xy_max[0] - cfg.region_of_interest_xy_min[0]) // ((cfg.subset_size + cfg.subset_offset) / 2))
subset_number_y = int((cfg.region_of_interest_xy_max[1] - cfg.region_of_interest_xy_min[1]) // ((cfg.subset_size + cfg.subset_offset) / 2))
print("Subset number_x: ", subset_number_x, ", number_y: ", subset_number_y)



# TODO:
crack_xy_min = [0, 0]
crack_xy_max = [0, 0]
#



reference_time = 0
initital_z_guess = 0
analyze_result = np.zeros((cfg.sub_group_size * subset_number_x, cfg.sub_group_size * subset_number_y, len(range(0, image_z_max, cfg.z_bounce)), 3))

def search_crack() :
    subsets_grid = np.zeros((int(cfg.image_x_max // ((cfg.subset_size + cfg.subset_offset) // 2)), int(cfg.image_y_max // ((cfg.subset_size + cfg.subset_offset) // 2))))

    # img = image_stacks_t[cfg.search_crack_asset_id].__getitem__(cfg.search_crack_img_id)

    img_path = cfg.assets[cfg.search_crack_asset_id] + cfg.search_crack_img_name
    im = (imread(img_path))
    grads = filters.sobel(im)
    grads_mean = grads.mean()

    threshold = 0.5

    for i in range (subsets_grid.shape[0]) :
        for j in range (subsets_grid.shape[1]) :
            subset_coord = np.array([i , j]) * ((cfg.subset_size + cfg.subset_offset) // 2)
            subset_grads = grads[subset_coord[0] : subset_coord[0] + cfg.subset_size, subset_coord[1] : subset_coord[1] + cfg.subset_size]
            if (subset_grads.mean() < grads_mean * threshold) :
                subsets_grid[i, j] = None # it's a crack

    return subsets_grid


subsets_gradients_grid = search_crack()
print(subsets_gradients_grid)

def is_subset_in_crack(xy_min, gradients_grid) :
    sz = np.array(gradients_grid.shape) - 1
    subset_grid_xy = xy_min // ((cfg.subset_size + cfg.subset_offset) // 2)
    return np.isnan(gradients_grid[int(subset_grid_xy[1]), int(subset_grid_xy[0])])


mesher = dic.MyMesher(type='q4')
for i in range (0, subset_number_x) :
    for j in range (0, subset_number_y):
        subset_center = cfg.region_of_interest_xy_min + (cfg.subset_size / 2 + cfg.subset_offset * np.array([i, j]))
        s_xy_min = subset_center - cfg.subset_size / 2
        s_xy_max = subset_center + cfg.subset_size / 2

        if (is_subset_in_crack(s_xy_min, subsets_gradients_grid)) :
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
            for z in range (search_z_min, search_z_max) :
                img_stack_tmp = dic.image_stack_from_list([image_stacks_t[reference_time].__getitem__(k), image_stacks_t[reference_time + 1].__getitem__(z)])
                mesh = mesher.my_mesh(img_stack_tmp, Xc1=s_xy_min[0], Xc2=s_xy_max[0], Yc1=s_xy_min[1], Yc2=s_xy_max[1], n_elx=cfg.sub_group_size, n_ely=cfg.sub_group_size)

                inputs = dic.DICInput(mesh,img_stack_tmp)
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

            uvz = np.array([u.flatten(), v.flatten(), np.full(cfg.sub_group_size ** 2, best_z - z)]).T.reshape(cfg.sub_group_size, cfg.sub_group_size, 3)

            analyze_result[cfg.sub_group_size*i : cfg.sub_group_size*(i+1), cfg.sub_group_size*j : cfg.sub_group_size*(j+1), k // cfg.z_bounce] = uvz

            #TODO :
            # initital_z_guess = ...
            #
            #         

print(subsets_gradients_grid)

bounced_z_img_id = cfg.show_img_id // cfg.z_bounce

xs = np.arange(cfg.region_of_interest_xy_min[0], 1 + cfg.region_of_interest_xy_max[0] - cfg.subset_size, cfg.subset_offset) + cfg.subset_size // 2
ys = np.arange(cfg.region_of_interest_xy_min[1], 1 + cfg.region_of_interest_xy_max[1] - cfg.subset_size, cfg.subset_offset) + cfg.subset_size // 2
plot_z = analyze_result[:, :, bounced_z_img_id, 2]

print("PLOT Z:", plot_z)
print("xs, ys:", xs, ys)

plt.imshow(image_stacks_z[cfg.show_img_id].__getitem__(0), cmap=plt.cm.gray, origin="lower", extent=(0, cfg.image_x_max, 0, cfg.image_y_max))
plt.pcolormesh(xs, ys, plot_z.T, alpha=0.3)
plt.colorbar()
plt.show()
