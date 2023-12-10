import logging
import glob

import muDIC as dic

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
                 roi_xy_min = None, roi_xy_max = None, maxit = 40) :
    
        self.logger = logging.getLogger()

        if subset_offset < subset_size:
            raise ValueError("Offset between subsets must be bigger or equal than subset size!")

        self.subset_size = subset_size
        self.subset_offset = subset_offset
        
        self.z_bounce = z_bounce
        self.z_radius = z_radius

        self.images_folder = images_folder
        self.image_name_prefix = image_name_prefix

        if (len(times_list) < 2) :
            raise ValueError("At least two image stacks are needed!")
        
        self.times_list = times_list
    
        self.maxit = maxit

        self.stack_h, self.image_stacks = self.__generate_image_stacks__()

        self.image_xy_min, self.image_xy_max = self.__get_image_bounds__()
        self.logger.info("Input images\' size is %ix%i" % (self.image_xy_max[1], self.image_xy_max[0]))

        if ((self.subset_size > self.image_xy_max[0]) or (self.subset_size > self.image_xy_max[1])) :
            raise ValueError("Subset size is bigger than image size!")

        if ((roi_xy_min is None) or (roi_xy_max is None)) :
            self.roi_xy_min, self.roi_xy_max = self.image_xy_min, self.image_xy_max
        else :
            self.roi_xy_min, self.roi_xy_max = roi_xy_min, roi_xy_max

        self.subset_number_y = int((self.roi_xy_max[0] - self.roi_xy_min[0] - self.subset_size) // (self.subset_offset)) + 1
        self.subset_number_x = int((self.roi_xy_max[1] - self.roi_xy_min[1] - self.subset_size) // (self.subset_offset)) + 1
        self.subset_number_z = len(range(0, self.stack_h, self.z_bounce))
    

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
        img = imread(self.images_folder + any_image_path)

        return np.array([0, 0]), np.array(img.shape)
    
    def __is_subset_in_crack__(self, xy_min, grads, crack_grad_threshold) :
        subset_grads = grads[xy_min[0] : xy_min[0] + self.subset_size, xy_min[1] : xy_min[1] + self.subset_size]
        if (subset_grads.mean() < crack_grad_threshold) :
            return True
        
        return False
    
    def __run_3D_DIC__(self, ref_stack, def_stack, crack_weight = 0.3) :
        s_x = self.subset_number_x
        s_y = self.subset_number_y
        s_z = self.subset_number_z

        subset_number = s_y * s_x
        sub_group_size = 1

        interesting_layers = range(0, self.stack_h, self.z_bounce)

        result_ref = np.zeros((s_z * subset_number, 3))
        result_def = np.zeros_like(result_ref)

        # Searching Crack in the image
        img_grads = filters.sobel(imread(self.images_folder + ref_stack[self.stack_h // 2]))
        crack_grad_threshold = crack_weight * img_grads.mean()

        mesher = dic.MyMesher(type='q4')
        for i in range (0, s_y) :
            for j in range (0, s_x):
                subset_center = self.roi_xy_min + (self.subset_size // 2 + self.subset_offset * np.array([i, j]))
                s_xy_min = subset_center - self.subset_size / 2
                s_xy_max = subset_center + self.subset_size / 2

                xyz_table_start = j * (s_y * s_z) + i * s_z

                if (self.__is_subset_in_crack__(s_xy_min.astype(int), img_grads, crack_grad_threshold)) :
                    result_ref[xyz_table_start:xyz_table_start + s_z] = np.concatenate((np.repeat(np.array([subset_center[1], subset_center[0]]), s_z, axis=1).T, interesting_layers[:, None]), axis = 1)
                    result_def[xyz_table_start:xyz_table_start + s_z] = np.array([None, None, None])
                    continue

                initital_z_guess = 0
                for k in interesting_layers :
                    k_bounced = k // self.z_bounce

                    result_ref[xyz_table_start + k_bounced] = np.array([subset_center[1], subset_center[0], k])

                    init_z = k + initital_z_guess
                    if (init_z < 0) :
                        init_z = 0
                    elif (init_z >= self.stack_h) :
                        init_z = self.stack_h - 1

                    search_z_min = init_z - self.z_radius
                    if (search_z_min < 0) :
                        search_z_min = 0
                    
                    search_z_max = init_z + self.z_radius
                    if (search_z_max >= self.stack_h) :
                        search_z_max = self.stack_h - 1

                    best_z = search_z_min
                    cur_koef = 1e5
                    best_results = None

                    ref_image_k = imread(self.images_folder + ref_stack[k])
                    mesh = mesher.my_mesh(dic.image_stack_from_list([ref_image_k]), Xc1=s_xy_min[1], Xc2=s_xy_max[1], Yc1=s_xy_min[0], Yc2=s_xy_max[0], n_elx=sub_group_size, n_ely=sub_group_size)

                    for z in range (search_z_min, search_z_max + 1) :             
                        img_stack_tmp = dic.image_stack_from_list([ref_image_k, imread(self.images_folder + def_stack[z])])

                        inputs = dic.DICInput(mesh, img_stack_tmp, maxit=self.maxit)
                        dic_job = dic.MyDICAnalysis(inputs)

                        results, koefs = dic_job.run()

                        if (koefs[0] < cur_koef) :
                            cur_koef = koefs[0]
                            best_z = z
                            best_results = results

                    fields = dic.Fields(best_results)
                    disps = fields.disp()
                    u = disps[0, 0, :, :, 1].squeeze()
                    v = disps[0, 1, :, :, 1].squeeze()

                    result_def[xyz_table_start + k_bounced] = np.array([subset_center[1] + v, subset_center[0] + u, best_z])
                    #TODO :
                    # initital_z_guess = ...
                    #
                    #
        
        return result_ref, result_def
    
    def run(self, output_folder = None) :

        self.logger.info("Subset number_x: %i, number_y: %i" % (self.subset_number_x, self.subset_number_y))

        total_result = np.zeros((len(self.image_stacks) - 1, 2, self.subset_number_x * self.subset_number_y * self.subset_number_z, 3))

        for i in range (0, len(self.image_stacks) - 1) :
            start_time = time.time()

            ref_stack = self.image_stacks[i]
            def_stack = self.image_stacks[i+1]
            total_result[i, 0], total_result[i, 1] = self.__run_3D_DIC__(ref_stack, def_stack)

            if (output_folder is not None) :
                np.save(output_folder + 'ref_' + str(self.times_list[i]) + '_' + str(self.times_list[i+1]) + '.npy', total_result[i, 0])
                np.save(output_folder + 'def_' + str(self.times_list[i]) + '_' + str(self.times_list[i+1]) + '.npy', total_result[i, 1])
            
            self.logger.info("Time spent for DIC between stack %i and %i is %f seconds." %(self.times_list[i], self.times_list[i+1], time.time() - start_time))

        return total_result
    
    def get_image_bounds(self) :
        return self.image_xy_max
    

         



