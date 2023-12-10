import volumeDIC.search as vlm

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


images_folder = r'images/demo/'
images_prefix = 'Crack_t'
output_folder = r'output/'
disps = os.listdir(output_folder)

subset_size, subset_offset = 400, 400
z_bounce = 2
z_radius = 1

times_list = [31, 40]
volumeSearcher = vlm.Searcher(subset_size, subset_offset, z_bounce, z_radius, images_folder, images_prefix, times_list)

if (len(disps) == 0) :
    total_results = volumeSearcher.run(output_folder)

result_ref = np.load(output_folder + 'ref_31_40.npy')
result_def = np.load(output_folder + 'def_31_40.npy')

# Let's show z disps for z = 2 for w = 31
time_to_show = times_list[0]
slice_index_to_show = 2 # z030
bounced_z_id = slice_index_to_show // z_bounce

image_size = volumeSearcher.get_image_bounds()

xs = np.arange(0, 1 + image_size[1] - subset_size, subset_offset) + subset_size // 2
ys = np.arange(0, 1 + image_size[0] - subset_size, subset_offset) + subset_size // 2
plot_z = (result_def[:, 2] - result_ref[:, 2])[bounced_z_id::volumeSearcher.subset_number_z]
plot_z = plot_z.reshape(volumeSearcher.subset_number_x, volumeSearcher.subset_number_y)

print("PLOT Z:", plot_z)

background_image = plt.imread(images_folder + images_prefix + str(time_to_show) + '_z030.png')
plt.imshow(background_image, cmap=plt.cm.gray, origin="lower", extent=(0, image_size[1], 0, image_size[0]))
plt.pcolormesh(xs, ys, plot_z, alpha=0.3)
plt.colorbar()
plt.show()



