import logging
import glob

import muDIC as dic

import xarray as xr

import numpy as np
import os
import math  
import time

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters

class Searcher :
    def __init__(self, subset_size, subset_offset, \
                 z_bounce, z_radius, images_folder, image_name_prefix, times_list, \
                 roi_xy_min = None, roi_xy_max = None, maxit = 40, down_sampling_factor = 1) :
    
        self.logger = logging.getLogger()

        if subset_offset < subset_size:
            raise ValueError("Offset between subsets must be bigger or equal than subset size!")

        if (type(down_sampling_factor) is not int) or (down_sampling_factor < 1):
            raise TypeError("Down sampling factor has to be a positive integer")
        
        self.subset_size = int(subset_size // down_sampling_factor)
        self.subset_offset = int(subset_offset // down_sampling_factor)
        self.sub_group_size = 1 # not used now
        
        self.z_bounce = z_bounce  #distances between interesting layers
        self.z_radius = z_radius

        self.images_folder = images_folder
        self.image_name_prefix = image_name_prefix

        if (len(times_list) < 2) :
            raise ValueError("At least two image stacks are needed!")
        
        self.times_list = times_list # indices of stacks' time in input path
        self.maxit = maxit # Max iterations for DIC corellator

        self.stack_h, self.image_stacks = self.__generate_image_stacks__()

        self.down_sampling_factor = down_sampling_factor
        self.down_sampler = None

        self.image_xy_min, self.image_xy_max = self.__get_image_bounds__()

        self.logger.info("Input images\' size is %ix%i" % (self.image_xy_max[1], self.image_xy_max[0]))

        if ((self.subset_size > self.image_xy_max[0]) or (self.subset_size > self.image_xy_max[1])) :
            raise ValueError("Subset size is bigger than image size!")

        if ((roi_xy_min is None) or (roi_xy_max is None)) :
            self.roi_xy_min, self.roi_xy_max = self.image_xy_min, self.image_xy_max
        else :
            self.roi_xy_min, self.roi_xy_max = int(roi_xy_min // down_sampling_factor), int(roi_xy_max // down_sampling_factor)

        self.subset_number_y = int((self.roi_xy_max[0] - self.roi_xy_min[0] - self.subset_size) // (self.subset_offset)) + 1
        self.subset_number_x = int((self.roi_xy_max[1] - self.roi_xy_min[1] - self.subset_size) // (self.subset_offset)) + 1
        self.subset_number_z = len(range(0, self.stack_h, self.z_bounce))

        self.mesher = dic.MyMesher(type='q4')
  

    def __generate_image_stacks__(self):
        self.logger.info("%i images found in \"%s\"" % (len(os.listdir(self.images_folder)), self.images_folder))

        image_stacks = []
        stack_h = None

        for i in self.times_list :
                stack = [os.path.basename(f) for f in glob.glob(self.images_folder + self.image_name_prefix + str(i) + '_z*.png', recursive=True)]
                stack_len = len(stack)
                if (stack_len == 0) :
                    raise ValueError("Image stack w%i has zero images!", i)
                
                if (stack_h == None) :
                    stack_h = stack_len
                elif (stack_h != stack_len) :
                    raise ValueError("All image stacks must have the same height!")
                
                stack.sort()
                image_stacks.append(stack)

        return stack_h, image_stacks
    
    def __get_image_bounds__(self) :
        any_image_path = self.image_stacks[0][0]
        img = self.read_image(self.images_folder + any_image_path)

        return np.array([0, 0]), np.array(img.shape)
    
    def __is_subset_in_crack__(self, xy_min, grads, crack_grad_threshold) :
        subset_grads = grads[xy_min[0] : xy_min[0] + self.subset_size, xy_min[1] : xy_min[1] + self.subset_size]
        if (subset_grads.mean() < crack_grad_threshold) :
            return True
        
        return False
    
    def __run_3D_DIC__(self, ref_stack, def_images, crack_weight = 0.3) :
        s_x = self.subset_number_x
        s_y = self.subset_number_y
        s_z = self.subset_number_z

        subset_number = s_y * s_x

        interesting_layers = range(0, self.stack_h, self.z_bounce)

        result_ref = np.zeros((s_z * subset_number, 3))
        result_def = np.zeros_like(result_ref)
        result_koefs = np.zeros((s_z * subset_number))

        # Searching Crack in the image
        img_grads = filters.sobel(self.read_image(self.images_folder + ref_stack[self.stack_h // 2]))
        crack_grad_threshold = crack_weight * img_grads.mean()

        initital_z_guess = np.array(interesting_layers, dtype=int)
        for i in reversed(range(0, s_y)) :
            row_range = range(0, s_x)
            if (i % 2 == 0) :
                row_range = reversed(row_range)
            for j in row_range:
                subset_center = self.roi_xy_min + (self.subset_size // 2 + self.subset_offset * np.array([i, j]))
                s_xy_min = subset_center - self.subset_size / 2
                s_xy_max = subset_center + self.subset_size / 2

                xyz_table_start = j * (s_y * s_z) + i * s_z

                if (self.__is_subset_in_crack__(s_xy_min.astype(int), img_grads, crack_grad_threshold)) :
                    result_ref[xyz_table_start:xyz_table_start + s_z] = np.concatenate((np.repeat(np.array([subset_center[1], subset_center[0]])[:, None], s_z, axis=1).T, np.array(interesting_layers)[:, None]), axis = 1)
                    result_def[xyz_table_start:xyz_table_start + s_z] = result_ref[xyz_table_start:xyz_table_start + s_z] #np.array([None, None, None])
                    result_koefs[xyz_table_start + k_bounced] = None
                    continue

                for k in interesting_layers :
                    ref_image = self.read_image(self.images_folder + ref_stack[k])
                    mesh = self.mesher.my_mesh(dic.image_stack_from_list([ref_image]), Xc1=s_xy_min[1], Xc2=s_xy_max[1], Yc1=s_xy_min[0], Yc2=s_xy_max[0], n_elx=self.sub_group_size, n_ely=self.sub_group_size)

                    k_bounced = int(k // self.z_bounce)

                    result_ref[xyz_table_start + k_bounced] = np.array([subset_center[1], subset_center[0], k])

                    init_z = initital_z_guess[k_bounced]

                    search_z_min = k - self.z_radius
                    if (search_z_min < 0) :
                        search_z_min = 0
                    
                    search_z_max = k + self.z_radius
                    if (search_z_max >= self.stack_h) :
                        search_z_max = self.stack_h - 1

                    best_koef = 1e6
                    for z in range (init_z, search_z_max + 1) :
                        u, v, koef = self.__calculate_layers_disps___(ref_image, def_images[z], mesh)
                        if (koef < best_koef) :
                            best_u, best_v, best_koef = u, v, koef
                            best_z = z
                        else :
                            break
                    
                    if (best_z > init_z) :
                        result_def[xyz_table_start + k_bounced] = np.array([subset_center[1] + best_v, subset_center[0] + best_u, best_z])
                        result_koefs[xyz_table_start + k_bounced] = best_koef

                        initital_z_guess[k_bounced] = best_z
                        continue

                    for z in reversed(range(search_z_min ,init_z)) :
                        u, v, koef = self.__calculate_layers_disps___(ref_image, def_images[z], mesh)
                        if (koef < best_koef) :
                            best_u, best_v, best_koef = u, v, koef
                            best_z = z
                        else :
                            break

                    result_def[xyz_table_start + k_bounced]   = np.array([subset_center[1] + best_v, subset_center[0] + best_u, best_z])
                    result_koefs[xyz_table_start + k_bounced] = best_koef

                    initital_z_guess[k_bounced] = best_z
        
        return result_ref, result_def, result_koefs
    
    def run(self, output_folder = None) :

        self.logger.info("Subset number_x: %i, number_y: %i, number_z: %i" % (self.subset_number_x, self.subset_number_y, self.subset_number_z))

        total_result = np.zeros((len(self.image_stacks) - 1, 2, self.subset_number_x * self.subset_number_y * self.subset_number_z, 3))
        corr_koefs = np.zeros((len(self.image_stacks) - 1, self.subset_number_x * self.subset_number_y * self.subset_number_z))

        for i in range (0, len(self.image_stacks) - 1) :
            start_time = time.time()

            ref_stack = self.image_stacks[i]
            def_stack = self.image_stacks[i+1]

            # reading deformed stack
            def_paths = [self.images_folder + def_image_name for def_image_name in def_stack]
            def_images = list(map(self.read_image, def_paths))

            total_result[i, 0], total_result[i, 1], corr_koefs[i] = self.__run_3D_DIC__(ref_stack, def_images)

            total_result[i, :, :, 0:2] *= self.down_sampling_factor # multiplying x and y because of downsampling

            if (output_folder is not None) :
                np.save(output_folder + 'ref_'  + str(self.times_list[i]) + '_' + str(self.times_list[i+1]) + '.npy', total_result[i, 0])
                np.save(output_folder + 'def_'  + str(self.times_list[i]) + '_' + str(self.times_list[i+1]) + '.npy', total_result[i, 1])
                np.save(output_folder + 'corr_' + str(self.times_list[i]) + '_' + str(self.times_list[i+1]) + '.npy', corr_koefs[i])
            
            self.logger.info("Time spent for DIC between stack %i and %i is %f seconds." %(self.times_list[i], self.times_list[i+1], time.time() - start_time))

        return total_result, corr_koefs
    
    def get_image_bounds(self) :
        return self.image_xy_max
    
    def read_image(self, path) :
        image = imread(path)
        if (self.down_sampling_factor != 1) :
            a = xr.DataArray(image, dims=['x', 'y'])
            image = a.coarsen(x=self.down_sampling_factor, y=self.down_sampling_factor).mean().to_numpy()

        return image
    
    def __calculate_layers_disps___(self, ref_image, def_image, mesh) :
        img_stack_k_init = dic.image_stack_from_list([ref_image, def_image])

        inputs = dic.DICInput(mesh, img_stack_k_init, maxit=self.maxit)
        dic_job = dic.MyDICAnalysis(inputs)

        best_results, koefs = dic_job.run()

        fields = dic.Fields(best_results)
        disps = fields.disp()
        u = disps[0, 0, :, :, 1].squeeze()
        v = disps[0, 1, :, :, 1].squeeze()

        return u, v, koefs[0]
        

    

         



