import volumeDIC.search as vlm

import numpy as np
import os
import matplotlib.pyplot as plt


images_folder = r'Crack_PNG_re-ordered/'
images_prefix = 'w'
output_folder = r'output/all_disps_200_no_downsampling/'
disps = os.listdir(output_folder)

subset_size, subset_offset = 200, 500
z_bounce = 20
z_radius = 3

times_list = range(0, 20)
volumeSearcher = vlm.Searcher(subset_size, subset_offset, z_bounce, z_radius, images_folder, images_prefix, times_list, down_sampling_factor=1, maxit=100)

if (len(disps) == 0) :
    total_results, corr_koefs = volumeSearcher.run(output_folder)

result_ref = np.load(output_folder + 'ref_0_1.npy')
result_def = np.load(output_folder + 'def_0_1.npy')
result_koefs = np.load(output_folder + 'corr_0_1.npy')
print(result_koefs)


time_to_show = times_list[3]
slice_index_to_show = 60 # z040
bounced_z_id = slice_index_to_show // z_bounce

image_size = volumeSearcher.get_image_bounds()

xs = np.arange(0, 1 + image_size[1] - volumeSearcher.subset_size, volumeSearcher.subset_offset) + volumeSearcher.subset_size // 2
ys = np.arange(0, 1 + image_size[0] - volumeSearcher.subset_size, volumeSearcher.subset_offset) + volumeSearcher.subset_size // 2
plot_z = (result_def[:, 2] - result_ref[:, 2])[bounced_z_id::volumeSearcher.subset_number_z]
plot_z = plot_z.reshape(volumeSearcher.subset_number_x, volumeSearcher.subset_number_y)

print("PLOT Z:", plot_z)

background_image = volumeSearcher.read_image(images_folder + images_prefix + str(time_to_show) + '_z060.png')
plt.imshow(background_image, cmap=plt.cm.gray, extent=(0, image_size[1], 0, image_size[0]))
plt.pcolormesh(xs, ys, plot_z, alpha=0.3)
plt.colorbar()
plt.show()



