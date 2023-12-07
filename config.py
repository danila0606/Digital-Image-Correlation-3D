import numpy as np

class Config :
    def __init__(self) :
        # crack_paths = [r"./images/Crack_30/", r"./images/Crack_31/", r"./images/Crack_40/"]
        self.assets = [r"./images/Crack_31_demo/", r"./images/Crack_40_demo/"]
        self.image_type = ".png"
        self.image_times = [31, 40]
        self.image_t_max = len(self.image_times) - 1
        
        self.search_crack_asset_id = 0 
        self.search_crack_img_id = 0  # img to seacrh crack area using mean gradient dencity
        self.search_crack_img_name = 'Crack_t31_z030.png'

        self.subset_size = 140
        self.subset_offset = 200

        self.image_x_max = 2048
        self.image_y_max = 2048

        self.region_of_interest_xy_min = np.array([0, 0])
        self.region_of_interest_xy_max = np.array([700, 700])

        self.z_bounce = 2
        self.z_radius = 1
        self.sub_group_size = 1

        self.show_img_id = 2

        #max iterations for DIC correlation minimization
        self.maxit = 40