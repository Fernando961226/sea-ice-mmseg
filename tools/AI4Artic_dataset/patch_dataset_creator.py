
"""
File use to create the patches for the ready2train version from AI4Artic dataset
"""


import argparse
import glob
import os
from icecream import ic
from tqdm import tqdm

import numpy as np
import torch.utils.data as data
import shutil
import joblib
import xarray as xr
import torch
import matplotlib.pyplot as plt
import re
import datetime
from dateutil import relativedelta

from parallel_stuff import Parallel


def Arguments():
    """
    Parses command-line arguments.

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/media/fernando/Databases/ai4arcticready2train_v2', type=str, help='')
    parser.add_argument('--downsampling', default=1, type=int, help='Downsampling of the scene')
    parser.add_argument('--patch_size', default=224, type=int, help='size of patch')
    parser.add_argument('--overlap', default=0.0, type=float, help='Amount of overlap. Max 1, Min 0')
    parser.add_argument('--output', default='/home/fernando/scratch/train', type=str, help='')
    args = parser.parse_args()
    return args


class Slide_patches_index(data.Dataset):
    '''
    A PyTorch Dataset class that generates sliding patches indexes from an image, 
    considering overlap and landmask. This class creates patches indexes of the image 
    and excludes patches that are fully within the masked area.
    
    Attributes:
        h_crop (int): Height of the crop (patch).
        w_crop (int): Width of the crop (patch).
        h_stride (int): Vertical stride between patches.
        w_stride (int): Horizontal stride between patches.
        h_grids (int): Number of vertical grids (patch positions).
        w_grids (int): Number of horizontal grids (patch positions).
        patches_list (list): List of valid patches defined by their coordinates.
    '''
    def __init__(self, h_img, w_img, patch_size, downscaling, overlap_percent, landmask):
        '''
        Initializes the Slide_patches_index with the given image dimensions, patch size, 
        overlap percentage, and landmask.
        
        Args:
            h_img (int): Height of the input image.
            w_img (int): Width of the input image.
            patch_size (int): Size of each patch (square patch).
            overlap_percent (float): Percentage of overlap between patches.
            landmask (numpy array): Boolean array indicating valid areas (land) in the image.
        '''
        super(Slide_patches_index, self).__init__()

        # calculate the new image size based on down scaling
        h_img_d, w_img_d = int(np.round(h_img/downscaling)), int(np.round(w_img/downscaling))

        # calculate the actual down scaling factor due to rounding
        d_h_img, d_w_img = h_img/h_img_d, w_img/w_img_d

        self.h_crop = patch_size if patch_size < h_img_d else h_img_d
        self.w_crop = patch_size if patch_size < w_img_d else w_img_d

        self.h_stride = self.h_crop - round(self.h_crop * overlap_percent) if self.h_crop < h_img_d else h_img_d
        self.w_stride = self.w_crop - round(self.w_crop * overlap_percent) if self.w_crop < w_img_d else w_img_d

        # Creates the number of grids
        self.h_grids = max(h_img_d - self.h_crop + self.h_stride - 1, 0) // self.h_stride + 1
        self.w_grids = max(w_img_d - self.w_crop + self.w_stride - 1, 0) // self.w_stride + 1

        self.patches_list = []

        # creates the indexes for each patch
        for h_idx in range(self.h_grids):
            for w_idx in range(self.w_grids):
                y1 = h_idx * self.h_stride
                x1 = w_idx * self.w_stride

                y2 = min(y1 + self.h_crop, h_img_d)
                x2 = min(x1 + self.w_crop, w_img_d)

                # TODO: Why this lines?
                y1 = max(y2 - self.h_crop, 0)
                x1 = max(x2 - self.w_crop, 0)

                # indexes for land since it is not down_scale
                y1_land = int(np.round(y1 * d_h_img))
                x1_land = int(np.round(x1 * d_w_img))

                y2_land = int(np.round(y2 * d_h_img))
                x2_land = int(np.round(x2 * d_w_img))

                # Removes the patches that are in land
                if not landmask[y1_land:y2_land, x1_land:x2_land].any():
                    self.patches_list.append((y1, y2, x1, x2))
                
                # program to verify the patches are working
                # print(f'Land mask:({x1_land},{y1_land}), ({y2_land},{x2_land})')
                # plt.imshow(landmask[y1_land:y2_land, x1_land:x2_land])
                # plt.title('landmask')
                # plt.show()
                

                # landmask_small = torch.nn.functional.interpolate(input=torch.from_numpy(np.float32(landmask)).view((1, 1, h_img, w_img)), size=(h_img_d, w_img_d), mode='bilinear').numpy().squeeze()
                # print(f'Small Land mask:({x1},{y1}), ({y2},{x2})')
                # plt.imshow(landmask_small[y1:y2, x1:x2])
                # plt.title('small landmask')
                # plt.show()

    def __getitem__(self, index):

        """
        Returns the image, ground truth, and background patches at the specified index.
        
        Args:
            index (int): Index of the patch.
        
        Returns:
            tuple: Image patch, ground truth patch, background patch.
        """
        return self.patches_list[index]
    
    def __len__(self):
        return len(self.patches_list)


def get_time_of_year(file_name):
    """
    Extracts the data as a month and days of the year from the file name.
    
    Args:
        file_name (str): The file name containing the date information.
    
    Returns:
        tuple: The month and the number of days since the beginning of the year.
    """
    pattern = re.compile(r'\d{8}T\d{6}')

    # Search for the first match in the string
    match = re.search(pattern, file_name)

    first_date = match.group(0)

    # parse the date string into a datetime object
    date2 = datetime.datetime.strptime(first_date, "%Y%m%dT%H%M%S")

    date1  = datetime.datetime(date2.year, 1, 1)
    # calculate the number of days between January 1st and the given date

    delta = relativedelta.relativedelta(date2, date1)

    months = delta.months
    days = (date2-date1).days
    return months, days


def Extract_patches(args, item):
    """
    Extracts patches from the scene file and saves them.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        item (tuple): Tuple containing the scene file and patch indexes.
    """
    
    scene_file, patch_idx = item
    down_scale = args.downsampling
    scene = xr.open_dataset(f, engine='h5netcdf')
    output_folder = os.path.join(args.output, os.path.split(scene_file)[1][:-3]+'_down_scale_'+str(down_scale)+'X')
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    ic(output_folder)
    data = {}

    # ----------- DOWNN SCALE SAR ------------- #

    rows, cols = scene['nersc_sar_primary'].shape

    rows_down, cols_down = int(np.round(rows/down_scale)), int(np.round(cols/down_scale))

    down_rows, down_cols = rows/rows_down, cols/cols_down


    if down_scale != 1:

        data['nersc_sar_primary'] = torch.nn.functional.interpolate(input=torch.from_numpy(scene['nersc_sar_primary'].values).view((1, 1, rows, cols)), 
                                                            size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()
    else:
        data['nersc_sar_primary'] = scene['nersc_sar_primary'].values


    # ----------- INTERPOLATE VARIABLES TO MATCH SAR SCALE ------------ #

    exclude_variables = ['nersc_sar_primary', 'SIC', 'FLOE', 'SOD']

    vars = list(scene.keys())
    filtered_vars = [var for var in vars if var not in exclude_variables]

    for var in filtered_vars:
        r, c = scene[var].shape
        data[var] = torch.nn.functional.interpolate(input=torch.from_numpy(scene[var].values).view((1, 1, r, c)), 
                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()

    sea_ice_maps = ['SIC', 'FLOE', 'SOD']
    for var in sea_ice_maps:
        r, c = scene[var].shape
        data[var] = torch.nn.functional.interpolate(input=torch.from_numpy(scene[var].values).view((1, 1, r, c)), 
                                                    size=(rows_down, cols_down), mode='nearest').numpy().squeeze()
                                            
    # -----------  PATCH EXTRACTION -------------- #
    data_patch = {}
    for i in range(len(patch_idx)):
        
        y1, y2, x1, x2 = patch_idx[i]

        for var in data.keys():
            data_patch[var] = data[var][y1:y2, x1:x2]

            if args.patch_size > np.abs(y1-y2):
                data_patch[var] = np.pad(data_patch[var], (0, args.patch_size - np.abs(y1-y2), 0, 0), 'symmetric')

            if args.patch_size > np.abs(x1-x2):
                data_patch[var] = np.pad(data_patch[var], (0, 0, 0, args.patch_size - np.abs(x1-x2)), 'symmetric')

        data_patch['scene_id'] = scene.attrs['scene_id']
        data_patch['indexes'] = [(x1*down_cols, y1*down_cols), (x2*down_cols, y2*down_rows)]
        data_patch['pixel_spacing'] = scene.attrs['pixel_spacing'] * down_scale
        data_patch['ice_service'] = scene.attrs['ice_service']
        months, days = get_time_of_year(scene.attrs['scene_id'])
        data_patch['month'] = months
        data_patch['day'] = days
        joblib.dump(data_patch, output_folder + "/{:05d}.pkl".format(i))


if __name__ == '__main__':   
    args = Arguments()

    # Grab all .nc files from root as a string list
    scene_files = glob.glob(args.root + '/*.nc')[0:1]

    patches_idx = []
    for f in tqdm(scene_files, ncols=50):
        scene = xr.open_dataset(f, engine='h5netcdf')
        row, col = scene['nersc_sar_primary'].shape
        landmask = scene['nersc_sar_primary'] == 0
        patches_idx.append(Slide_patches_index(row, col, args.patch_size, args.downsampling, args.overlap, landmask))
    
    iterable = zip(scene_files, patches_idx)
    
    if len(scene_files) > 1:
        Parallel(Extract_patches, iterable, args)
    else:
        Extract_patches(args, next(iterable))    
    

