import volumeDIC.search as vlm

import numpy as np
import os
import matplotlib.pyplot as plt

from volumeDIC.drawer import draw_image_disps

images_folder = r'Crack_PNG_re-ordered/'
images_prefix = 'w'
output_folder = r'output/all_disps/'
disps = os.listdir(output_folder)

subset_size, subset_offset = 100, 100
z_bounce = 20
z_radius = 3

times_list = range(10, 20)
volumeSearcher = vlm.Searcher(subset_size, subset_offset, z_bounce, z_radius, images_folder, images_prefix, times_list, down_sampling_factor=1)

if (len(disps) == 0) :
    total_results, corr_koefs = volumeSearcher.run(output_folder)

result_ref = np.load(output_folder + 'ref_2_3.npy')
result_def = np.load(output_folder + 'def_2_3.npy')
result_koefs = np.load(output_folder + 'corr_2_3.npy')

time_to_show = 3
slice_index_to_show = 40 # z040
show_image_path = images_folder + images_prefix + str(time_to_show) + '_z040.png'
show_disp = 'uv' # 'z'

bounced_z_id = slice_index_to_show // z_bounce
s_x, s_y, s_z = volumeSearcher.subset_number_x, volumeSearcher.subset_number_y, volumeSearcher.subset_number_z
points_ref, points_def = result_ref[bounced_z_id::s_z], result_def[bounced_z_id::s_z]

if (show_disp == 'uv') :
    draw_image_disps(show_image_path, points_ref[:, :2], points_def[:, :2])
else :

    disp = (points_def - points_ref)

    image_size = volumeSearcher.get_image_bounds()
    xs = np.arange(0, 1 + image_size[1] - volumeSearcher.subset_size, volumeSearcher.subset_offset) + volumeSearcher.subset_size // 2
    ys = np.arange(0, 1 + image_size[0] - volumeSearcher.subset_size, volumeSearcher.subset_offset) + volumeSearcher.subset_size // 2

    background_image = volumeSearcher.read_image(show_image_path)
    plt.imshow(background_image, cmap=plt.cm.gray, extent=(0, image_size[1], image_size[0], 0))
    plt.pcolormesh(xs, ys, disp[:, 2].reshape(s_x, s_y), alpha=0.5)
    plt.colorbar()
    plt.show()






