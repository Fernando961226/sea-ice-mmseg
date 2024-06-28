"""
File use to create the patches for the raw data from AI4Artic
"""

import time
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
from convert_raw_icechart import convert_polygon_icechart
from parallel_stuff import Parallel
from scipy.interpolate import RegularGridInterpolator
import wandb


def Arguments():
    """
    Parses command-line arguments.

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/' + os.getenv('LOGNAME') + '/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3', type=str, help='')
    parser.add_argument('--output', default='/home/' + os.getenv('LOGNAME') + '/scratch/dataset/ai4arctic/', type=str, help='')

    parser.add_argument('--downsampling', default=3, type=int, help='Downsampling of the scene')
    parser.add_argument('--patch_size', default=256, type=int, help='size of patch')
    parser.add_argument('--overlap', default=0.0, type=float, help='Amount of overlap. Max 1, Min 0')

    parser.add_argument('--n_cores', default=16, type=int, help='Number of CPU cores to use in parallel process')
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
    def __init__(self, h_img, w_img, patch_size, downsampling, overlap_percent, nan_mask):
        '''
        Initializes the Slide_patches_index with the given image dimensions, patch size, 
        overlap percentage.
        
        Args:
            h_img (int): Height of the input image.
            w_img (int): Width of the input image.
            patch_size (int): Size of each patch (square patch).
            overlap_percent (float): Percentage of overlap between patches.
            nan_mask (boolean array): If true then that area is NaN.
        '''
        super(Slide_patches_index, self).__init__()

        # calculate the new image size based on down scaling
        # if we downscale, we first apply a 2x2 block average
        if downsampling > 1:
            h_img_d, w_img_d = int(h_img//2//(downsampling/2)), int(w_img//2//(downsampling/2))
        else:
            h_img_d, w_img_d = h_img, w_img

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

                # This line guarantees complete patches at the last row/column of the patching grid
                y1 = max(y2 - self.h_crop, 0)
                x1 = max(x2 - self.w_crop, 0)

                # indexes for land since it is not down_scale
                y1_nan = int(np.round(y1 * d_h_img))
                x1_nan = int(np.round(x1 * d_w_img))

                y2_nan = int(np.round(y2 * d_h_img))
                x2_nan = int(np.round(x2 * d_w_img))

                # Removes the patches that are in land
                if not nan_mask[y1_nan:y2_nan, x1_nan:x2_nan].any():
                    self.patches_list.append((y1, y2, x1, x2))
                
                # program to verify the patches are working
                # print(f'Nan mask:({x1_nan},{y1_nan}), ({y2_nan},{x2_nan})')
                # plt.imshow(landmask[y1_nan:y2_nan, x1_nan:x2_nan])
                # plt.title('landmask')
                # plt.show()
                

                # nan_mask_small = torch.nn.functional.interpolate(input=torch.from_numpy(np.float32(nan_mask)).view((1, 1, h_img, w_img)), size=(h_img_d, w_img_d), mode='bilinear').numpy().squeeze()
                # print(f'Small nan mask:({x1},{y1}), ({y2},{x2})')
                # plt.imshow(nan_mask_small[y1:y2, x1:x2])
                # plt.title('Small Nan Mask')
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


def get_patch_index(args, scene_file):
    """
    Calculates the patches indexes
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        scene_file (str): String containing the scene file path.
    """
    
    scene = xr.open_dataset(scene_file, engine='h5netcdf')
    row, col = scene['nersc_sar_primary'].shape
    nan_mask = np.isnan(scene['nersc_sar_primary'])
    return Slide_patches_index(row, col, args.patch_size, args.downsampling, args.overlap, nan_mask)

def Extract_patches(args, item):
    """
    Extracts patches from the scene file and saves them.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        item (tuple): Tuple containing the scene file and patch indexes.
    """
    
    scene_file, patch_idx = item
    if not len(patch_idx):
        print("Number of patches = 0 for scene %s"%(os.path.split(scene_file)[1]))
        return
    
    down_scale = args.downsampling
    scene = xr.open_dataset(scene_file, engine='h5netcdf')
    output_folder = os.path.join(args.output, 'down_scale_'+str(down_scale)+'X', os.path.split(scene_file)[1][:-3])
    ic(output_folder)
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    data = {}

    #  ---------- Get the SIC, SOD, FLOE Charts
    scene = convert_polygon_icechart(scene)

    # ----------- GET SAR DATA
    data['nersc_sar_primary'] = scene['nersc_sar_primary'].values
    data['nersc_sar_secondary'] = scene['nersc_sar_secondary'].values
    rows, cols = scene['nersc_sar_primary'].shape

    # ----------- CREATE NAN MASK FROM SAR
    data['sar_nan_mask'] = np.isnan(data['nersc_sar_primary'])
    # data['nersc_sar_primary'][data['sar_nan_mask']] = 0
    # data['nersc_sar_secondary'][data['sar_nan_mask']] = 0

    # ----------- DOWNN SCALE SAR
    rows_down, cols_down = rows, cols

    if down_scale > 1:
        # Block average 2x2 before downsampling SAR channels
        data['nersc_sar_primary'] = torch.nn.functional.avg_pool2d(torch.from_numpy(scene['nersc_sar_primary'].values).unsqueeze(0).unsqueeze(0), 
                                                                   kernel_size = 2, stride = 2).squeeze().numpy()
        data['nersc_sar_secondary'] = torch.nn.functional.avg_pool2d(torch.from_numpy(scene['nersc_sar_secondary'].values).unsqueeze(0).unsqueeze(0),
                                                                   kernel_size = 2, stride = 2).squeeze().numpy()

        rows_down, cols_down = int(rows//2//(down_scale/2)), int(cols//2//(down_scale/2))
        down_rows, down_cols = rows/rows_down, cols/cols_down
        data['nersc_sar_primary'] = torch.nn.functional.interpolate(input=torch.from_numpy(scene['nersc_sar_primary'].values).unsqueeze(0).unsqueeze(0),
                                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()
        data['nersc_sar_secondary'] = torch.nn.functional.interpolate(input=torch.from_numpy(scene['nersc_sar_secondary'].values).unsqueeze(0).unsqueeze(0),
                                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()
        # Down scale SAR nan mask
        data['sar_nan_mask'] = torch.nn.functional.interpolate(input=torch.from_numpy(np.float32(data['sar_nan_mask'])).unsqueeze(0).unsqueeze(0),
                                                            size=(rows_down, cols_down), mode='nearest').numpy().squeeze().astype(bool)

    # ----------- INTERPOLATE GRID VARIABLES TO MATCH SAR

    grid_variables = ['sar_grid_latitude', 'sar_grid_longitude', 'sar_grid_incidenceangle']

    # Extract and reshape the initial x and y coordinates
    x = scene['sar_grid_sample'].values
    x_l = np.unique(x)

    y = scene['sar_grid_line'].values
    y_l = np.unique(y)

    # Define the finer grid for interpolation
    x_fine = np.linspace(0, cols - 1, cols_down)
    y_fine = np.linspace(0, rows - 1, rows_down)
    X, Y = np.meshgrid(x_fine, y_fine, indexing='xy')
    points = np.array([Y.flatten(), X.flatten()]).T

    # Loop through each variable to reshape, create interpolator, interpolate, and store results
    for var in grid_variables:
        values = scene[var].values
        reshaped_values = np.reshape(values, (len(y_l), len(x_l)))
        interpolator = RegularGridInterpolator((y_l, x_l), reshaped_values, method='linear')
        interpolated_values = interpolator(points)
        data[var] = interpolated_values.reshape(X.shape)

    # ----------- INTERPOLATE VARIABLES TO MATCH SAR SCALE

    exclude_variables = ['nersc_sar_primary', 'nersc_sar_secondary, ''polygon_icechart', 'sar_grid_line', 'sar_grid_sample', 
                         'sar_grid_latitude', 'sar_grid_longitude', 'sar_grid_incidenceangle', 'sar_grid_height', 
                         'amsr2_swath_map', 'swath_segmentation', 'SIC', 'FLOE', 'SOD']

    vars = list(scene.keys())
    filtered_vars = [var for var in vars if var not in exclude_variables]

    for var in filtered_vars:
        data[var] = torch.nn.functional.interpolate(input=torch.from_numpy(scene[var].values).unsqueeze(0).unsqueeze(0), 
                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()

    sea_ice_maps = ['SIC', 'FLOE', 'SOD']
    for var in sea_ice_maps:
        data[var] = torch.nn.functional.interpolate(input=torch.from_numpy(scene[var].values).unsqueeze(0).unsqueeze(0), 
                                                    size=(rows_down, cols_down), mode='nearest').numpy().squeeze()


    # ----------- PATCH EXTRACTION -------------- #
    data_patch = {}
    for i in range(len(patch_idx)):
        
        y1, y2, x1, x2 = patch_idx[i]

        for var in data.keys():
            data_patch[var] = data[var][y1:y2, x1:x2]

            if args.patch_size > np.abs(y1-y2):
                data_patch[var] = np.pad(data_patch[var], (0, args.patch_size - np.abs(y1-y2), 0, 0), 'symmetric')

            if args.patch_size > np.abs(x1-x2):
                data_patch[var] = np.pad(data_patch[var], (0, 0, 0, args.patch_size - np.abs(x1-x2)), 'symmetric')

        data_patch['file_name'] = os.path.split(scene_file)[1]
        data_patch['scene_id'] = data_patch['file_name'][17:32] + '_' + data_patch['file_name'][77:80]
        data_patch['indexes'] = [(x1*down_cols, y1*down_cols), (x2*down_cols, y2*down_rows)]
        data_patch['pixel_spacing'] = 40 * down_scale
        data_patch['ice_service'] = data_patch['file_name'][77:80]
        month, day = get_time_of_year(data_patch['scene_id'])
        data_patch['month'] = np.ones((args.patch_size, args.patch_size)) * month
        data_patch['day'] = np.ones((args.patch_size, args.patch_size)) * day
        joblib.dump(data_patch, output_folder + "/{:05d}.pkl".format(i))

if __name__ == '__main__':   

    # wandb.init(project='extract_patches')
    args = Arguments()

    # Grab all .nc files from root as a string list
    scene_files = glob.glob(args.root + '/*.nc')
    
    #  ---------------- GET INDEXES
    start_time = time.time()
    print('Calculating patches ...')
    iterable = iter(scene_files)
    if len(scene_files) > 1:
        patches_idx = Parallel(get_patch_index, iterable, args)
    else:
        patches_idx = [get_patch_index(args, next(iterable))]
    print('Patches index generated! - time:%.2f'%(time.time()-start_time))

    print('Number of Patches for downsampling %dX: %d'%(args.downsampling, sum([len(l) for l in patches_idx])))
    
    #  ---------------- EXTRACT PATCHES
    start_time = time.time()
    print('Saving patches ...')
    iterable = zip(scene_files, patches_idx)
    if len(scene_files) > 1:
        # Using more than 16 cores makes the memory consumption to explode
        Parallel(Extract_patches, iterable, args, n_cores=args.n_cores)
    else:
        Extract_patches(args, next(iterable))
    print('Patches saved! - time:%.2f'%(time.time()-start_time))
    

