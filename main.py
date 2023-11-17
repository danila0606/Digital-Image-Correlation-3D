import muDIC as dic

import numpy as np
import matplotlib.pyplot as plt


# crack_paths = [r"./images/Crack_30/", r"./images/Crack_31/", r"./images/Crack_40/"]

# CONFIG
crack_paths = [r"./images/Crack_31_demo/", r"./images/Crack_40_demo/"]
image_times = [31, 40]
image_t_max = len(image_times) - 1
#

image_stacks_t = []
image_z_max = int(1e10)

for path in crack_paths:
    image_stack = dic.image_stack_from_folder(path,file_type=".png") # t is the same
    image_z_max = min(image_z_max, image_stack.__len__())
    image_stacks_t.append(image_stack)

image_stacks_z = []

for i in range(0, image_z_max) :
    image_stack_z = [image_stacks_t[k][i] for k in range (0, image_t_max)]
    image_stacks_z.append(dic.image_stack_from_list(image_stack_z))


image_x_max, image_y_max = image_stacks_z[0].__getitem__(0).shape
print("Image size, x: ", image_x_max, ", y: ", image_y_max)


# CONFIG
subset_size = 140
subset_offset = 140

region_of_interest_xy_min = np.array([1200, 0])
region_of_interest_xy_max = np.array([image_x_max, image_y_max])

if subset_offset < subset_size:
    raise TypeError("Offset between subsets must be bigger or equal than subset size!")
#

subset_number_x = int((region_of_interest_xy_max[0] - region_of_interest_xy_min[0]) // ((subset_size + subset_offset) / 2))
subset_number_y = int((region_of_interest_xy_max[1] - region_of_interest_xy_min[1]) // ((subset_size + subset_offset) / 2))
print("Subset number_x: ", subset_number_x, ", number_y: ", subset_number_y)

# CONFIG
z_bounce = 2
z_radius = 1
sub_group_size = 1

    # TODO:
crack_xy_min = [0, 0]
crack_xy_max = [0, 0]
#



reference_time = 0
initital_z_guess = 0
analyze_result = np.zeros((sub_group_size * subset_number_x, sub_group_size * subset_number_y, len(range(0, image_z_max, z_bounce)), 3))
mesher = dic.MyMesher(type='q4')
for i in range (0, subset_number_x) :
    for j in range (0, subset_number_y):
        subset_center = region_of_interest_xy_min + np.array([subset_size / 2 + subset_offset * i, subset_size / 2 + subset_offset * j])
        s_xy_min = subset_center - subset_size / 2
        s_xy_max = subset_center + subset_size / 2

        for k in range(0, image_z_max, z_bounce) :
            init_z = k + initital_z_guess
            if (init_z < 0) :
                init_z = 0
            elif (init_z >= image_z_max) :
                init_z = image_z_max - 1

            search_z_min = init_z - z_radius
            if (search_z_min < 0) :
                search_z_min = 0
            
            search_z_max = init_z + z_radius
            if (search_z_max >= image_z_max) :
                search_z_max = image_z_max - 1

            best_z = search_z_min
            cur_koef = 1e5
            best_results = None
            for z in range (search_z_min, search_z_max + 1) :
                img_stack_tmp = dic.image_stack_from_list([image_stacks_t[reference_time].__getitem__(k), image_stacks_t[reference_time + 1].__getitem__(z)])
                mesh = mesher.my_mesh(img_stack_tmp, Xc1=s_xy_min[0], Xc2=s_xy_max[0], Yc1=s_xy_min[1], Yc2=s_xy_max[1], n_elx=sub_group_size, n_ely=sub_group_size)

                inputs = dic.DICInput(mesh,img_stack_tmp)
                dic_job = dic.MyDICAnalysis(inputs)

                results, koefs = dic_job.run()

                deformed_id = image_t_max - 1

                if (koefs[deformed_id] < cur_koef) :
                    cur_koef = koefs[deformed_id]
                    best_z = z
                    best_results = results

            fields = dic.Fields(best_results)
            disps = fields.disp()
            u = disps[0, 0, :, :, deformed_id]
            v = disps[0, 1, :, :, deformed_id]

            uvz = np.array([u.flatten(), v.flatten(), np.full(sub_group_size ** 2, best_z - z)]).T.reshape(sub_group_size, sub_group_size, 3)

            analyze_result[sub_group_size*i : sub_group_size*(i+1), sub_group_size*j : sub_group_size*(j+1), k % z_bounce] = uvz

            #TODO :
            # initital_z_guess = ...
            #

                
# CONFIG
show_img_id = 2
#

bounced_z_img_id = show_img_id % z_bounce

xs = np.arange(region_of_interest_xy_min[0], region_of_interest_xy_max[0] - subset_size, subset_offset) + subset_size // 2
ys = np.arange(region_of_interest_xy_min[1], region_of_interest_xy_max[1] - subset_size, subset_offset) + subset_size // 2
plot_z = analyze_result[:, :, bounced_z_img_id, 2]

print("PLOT Z:", plot_z)
print("xs, ys:", xs, ys)

plt.imshow(image_stacks_z[show_img_id].__getitem__(0), cmap=plt.cm.gray, origin="lower", extent=(0, image_x_max, 0, image_y_max))
plt.pcolormesh(xs, ys, plot_z.T, alpha=0.3)
plt.colorbar()
plt.show()


# print("Converted shape:", len(image_stacks_z))



# stack = image_stacks_z[4]
# print("Len:", stack.__len__())
# print("Type:", type(stack))

# mesh = mesher.mesh(stack)

# inputs = dic.DICInput(mesh, stack)

# dic_job = dic.DICAnalysis(inputs)
# results = dic_job.run()

# fields = dic.Fields(results)

# true_strain = fields.true_strain()

# viz = dic.Visualizer(fields,images=stack)
# print("Disp:", fields.disp())

# viz.show(field="disp", component = (0,1), frame = 1)