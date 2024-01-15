import logging
import glob

import numpy as np
import os
import sys 
import time
import cv2

from skimage import filters

class Searcher :
    def __init__(self, subset_size, subset_offset, \
                 z_bounce, z_radius, images_folder, image_name_prefix, times_list, \
                 roi_xy_min = None, roi_xy_max = None, down_sampling_factor = 1) :

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(100)
        self.logger = logging.getLogger()
        self.logger.addHandler(console)

        if subset_offset < subset_size:
            raise ValueError("Offset between subsets must be bigger or equal than subset size!")

        if (type(down_sampling_factor) is not int) or (down_sampling_factor < 1):
            raise TypeError("Down sampling factor has to be a positive integer")
        
        self.subset_size = int(subset_size // down_sampling_factor)
        self.subset_offset = int(subset_offset // down_sampling_factor)
        
        self.z_bounce = z_bounce  #distances between interesting layers
        self.z_radius = z_radius

        self.images_folder = images_folder
        self.image_name_prefix = image_name_prefix

        if (len(times_list) < 2) :
            raise ValueError("At least two image stacks are needed!")
        
        self.times_list = times_list # indices of stacks' time in input path
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
            self.roi_xy_min, self.roi_xy_max = (roi_xy_min // down_sampling_factor).astype(int), (roi_xy_max // down_sampling_factor).astype(int)

        self.subset_number_x = int((self.roi_xy_max[0] - self.roi_xy_min[0] - self.subset_size) // (self.subset_offset)) + 1
        self.subset_number_y = int((self.roi_xy_max[1] - self.roi_xy_min[1] - self.subset_size) // (self.subset_offset)) + 1
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
        img = self.read_image(self.images_folder + any_image_path)

        return np.array([0, 0]), np.array(img.T.shape)
    
    def __is_subset_in_crack__(self, xy_min, grads, crack_grad_threshold) :
        subset_grads = grads[xy_min[1] : xy_min[1] + self.subset_size, xy_min[0] : xy_min[0] + self.subset_size]
        # print(subset_grads.mean(), crack_grad_threshold)
        if (subset_grads.mean() < crack_grad_threshold) :
            return True
        
        return False
    
    def __run_3D_DIC__(self, ref_stack, def_images, crack_weight = 0.8) :
        s_x = self.subset_number_x
        s_y = self.subset_number_y
        s_z = self.subset_number_z

        subset_number = s_x * s_y

        interesting_layers = range(0, self.stack_h, self.z_bounce)

        result_ref = np.zeros((s_z * subset_number, 3))
        result_def = np.zeros_like(result_ref)
        result_koefs = np.full((s_z * subset_number), np.nan)

        # Searching Crack in the image
        img_grads = filters.sobel(self.read_image(self.images_folder + ref_stack[self.stack_h // 2]))
        crack_grad_threshold = crack_weight * img_grads.mean()

        initital_z_guess = np.array(interesting_layers, dtype=int)
        for i in (range(0, s_y)) :
            row_range = range(0, s_x)
            if (i % 2 == 0) :
                row_range = reversed(row_range)
            for j in row_range:
                s_xy_min = self.roi_xy_min + self.subset_offset * np.array([j, i])
                subset_center = (s_xy_min + self.subset_size / 2)

                xyz_table_start = i * (s_x * s_z) + j * s_z

                if (self.__is_subset_in_crack__(s_xy_min.astype(int), img_grads, crack_grad_threshold)) :
                    # putting undeformed coordinates
                    result_ref[xyz_table_start:xyz_table_start + s_z] = np.concatenate((np.repeat(np.array([subset_center[0], subset_center[1]])[:, None], s_z, axis=1).T, np.array(interesting_layers)[:, None]), axis = 1)
                    result_def[xyz_table_start:xyz_table_start + s_z] = result_ref[xyz_table_start:xyz_table_start + s_z]
                    continue

                for k in interesting_layers :
                    ref_image = self.read_image(self.images_folder + ref_stack[k])

                    k_bounced = int(k // self.z_bounce)

                    result_ref[xyz_table_start + k_bounced] = np.array([subset_center[0], subset_center[1], k])

                    init_z = initital_z_guess[k_bounced]

                    search_z_min = k - self.z_radius
                    if (search_z_min < 0) :
                        search_z_min = 0
                    
                    search_z_max = k + self.z_radius
                    if (search_z_max >= self.stack_h) :
                        search_z_max = self.stack_h - 1

                    best_koef = 1e6
                    best_u, best_v, best_z = 0., 0., init_z
                    for z in range (init_z, search_z_max + 1) :
                        u, v, koef = self.__calculate_layers_disps___(ref_image, def_images[z], s_xy_min)
                        if (koef < best_koef) :
                            best_u, best_v, best_koef = u, v, koef
                            best_z = z
                        else :
                            break
                    
                    if (best_z > init_z) :
                        result_def[xyz_table_start + k_bounced] = np.array([subset_center[0] + best_u, subset_center[1] + best_v, best_z])
                        result_koefs[xyz_table_start + k_bounced] = best_koef
                        initital_z_guess[k_bounced] = best_z
                        continue

                    for z in reversed(range(search_z_min ,init_z)) :
                        u, v, koef = self.__calculate_layers_disps___(ref_image, def_images[z], s_xy_min)
                        if (koef < best_koef) :
                            best_u, best_v, best_koef = u, v, koef
                            best_z = z
                        else :
                            break

                    result_def[xyz_table_start + k_bounced]   = np.array([subset_center[0] + best_u, subset_center[1] + best_v, best_z])
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
        image = cv2.imread(path, 0)
        if (self.down_sampling_factor != 1) :
            width = int(image.shape[1] / self.down_sampling_factor)
            height = int(image.shape[0] / self.down_sampling_factor)
            image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        return image
    
    def __calculate_layers_disps___(self, ref_image, def_image, s_xy_min) :

        s_size = np.array([np.float32(self.subset_size), np.float32(self.subset_size)])
        float_min = np.array([np.float32(s_xy_min[0]), np.float32(s_xy_min[1])])
        p_1 = float_min
        p_2 = np.array([p_1[0], p_1[1] + s_size[1]])
        p_3 = np.array([p_1[0] + s_size[0], p_1[1]])
        p_4 = p_1 + s_size

        points_in = np.array([p_1, p_2, p_3, p_4])

        lk_params = dict( winSize  = (self.subset_size, self.subset_size), maxLevel = 10,
											 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        points_out, st, err = cv2.calcOpticalFlowPyrLK(ref_image, def_image, points_in, None, **lk_params)

        if (np.sum(st) == 0) :
            return 0., 0., 1e6
        
        uv_s = points_out - points_in
        uv_mean = np.sum(uv_s, axis = 0) / np.sum(st)
        err_mean = np.sum(err) / sum(st)

        return uv_mean[0], uv_mean[1], err_mean
    
        

    

         



