'''
No@
'''
import xarray as xr
from mmcv.image import imread
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from icecream import ic
import os
import cv2
from tqdm import tqdm
import torch
import numpy as np
import joblib
from scipy.interpolate import RegularGridInterpolator
import re
from dateutil import relativedelta
import datetime

from AI4ArcticSeaIceChallenge.convert_raw_icechart import convert_polygon_icechart
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import glob
try:
    from osgeo import gdal
except ImportError:
    gdal = None

@TRANSFORMS.register_module()
class LoadPatchFromPKLFile(BaseTransform):
    """
        Load a single precomputed patch containing a dictionary with the channels:

        ## SAR
            -   nersc_sar_primary
            -   nersc_sar_secondary

        ## incidence angle
            -   sar_grid_incidenceangle

        ## Geographical variables 
            -   sar_grid_latitude
            -   sar_grid_longitude
            -   distance_map

        ## Environmental variables
            -   btemp_6_9h', 'btemp_6_9v',
            -   btemp_7_3h', 'btemp_7_3v',
            -   btemp_10_7h', 'btemp_10_7v',
            -   btemp_18_7h', 'btemp_18_7v',
            -   btemp_23_8h', 'btemp_23_8v',
            -   btemp_36_5h', 'btemp_36_5v',
            -   btemp_89_0h', 'btemp_89_0v',

        ## acquisition time
            -   month
            -   day

        Info
            -   file_name
            -   scene_id
            -   indexes
            -   pixel_spacing
            -   ice_service

        ## annotation 
            -   SIC
            -   FLOE
            -   SOD

       Required Keys:
           - img_path: Path to the directory containing the patch data.

       Modified Keys:
           - img: The loaded image data as a numpy array.
           - img_shape: The shape of the loaded image.
           - ori_shape: The original shape of the loaded image.
           - gt_seg_map (optional): The loaded segmentation map as a numpy array.

       Args:
           channels (list[str]): List of variable names to load as channels of the image #TODO: modify in config file.
           mean (dict{float}): Mean values for normalization of each channel. Defaults to values provided.
           std (dict{float}): Standard deviation values for normalization of each channel. Defaults to values provided.
           to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. Defaults to True.
           
           with_seg (bool): Whether to also load segmentation maps. Defaults to False.
           GT_type (list[str]): List of ground truth types to load (e.g., ['SOD', 'SIC', 'FLOE']). Defaults to ['SOD'].
       """

    def __init__(self,
                 channels,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 nan=255,
                 with_seg=False,
                 GT_type=['SOD']):

        self.channels = channels
        self.mean = mean
        self.std = std
        self.to_float32 = to_float32
        self.nan = nan
        self.with_seg = with_seg
        self.GT_type = GT_type


    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        data = joblib.load(filename)

        # Filter channels and normalize
        img_ = np.asarray([(data[key] - self.mean[key]) / self.std[key] for key in self.channels])
        img_ = img_.transpose((1, 2, 0))
        if self.to_float32:
            img = img_.astype(np.float32)
        img = np.nan_to_num(img, nan=self.nan)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        if self.with_seg:
            # Filter task
            seg_maps = np.asarray([data[key] for key in self.GT_type]).transpose((1, 2, 0))
            seg_maps = np.nan_to_num(seg_maps, nan=self.nan)        # NaN values in labels
            seg_maps[np.isnan(img_[:,:,0])] = self.nan              # NaN values in image
            results['gt_seg_map'] = seg_maps
            results['seg_fields'].append('gt_seg_map')

        return results

@TRANSFORMS.register_module()
class PreLoadImageandSegFromNetCDFFile(BaseTransform):
    """Pre-load images and segmentation maps from NetCDF files into memory.

       This transform pre-loads images and optionally segmentation maps from NetCDF files
       into memory to speed up the data loading process during training and inference.

       Required Keys:
           - img_path: Path to the NetCDF file containing the image data.

       Modified Keys:
           - img: The loaded image data as a numpy array.
           - img_shape: The shape of the loaded image.
           - ori_shape: The original shape of the loaded image.
           - gt_seg_map (optional): The loaded segmentation map as a numpy array.

       Args:
           channels (list[str]): List of variable names to load as channels of the image.
           data_root (str): Root directory of the NetCDF files.
           gt_root (str): Root directory of the ground truth segmentation maps.
           ann_file (str, optional): Path to the annotation file listing NetCDF files to load.
           mean (list[float]): Mean values for normalization of each channel. Defaults to values provided.
           std (list[float]): Standard deviation values for normalization of each channel. Defaults to values provided.
           to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. Defaults to True.
           color_type (str): The color type for image loading. Defaults to 'color'.
           imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
           nan (float): Value to replace NaNs in the image. Defaults to 255.
           downsample_factor (int): Factor by which to downsample the images. Defaults to 10.
           pad_size (tuple[int], optional): Desired size to pad the images to. Defaults to None.
           with_seg (bool): Whether to also load segmentation maps. Defaults to False.
           GT_type (list[str]): List of ground truth types to load (e.g., ['SOD', 'SIC', 'FLOE']). Defaults to ['SOD'].
           ignore_empty (bool): Whether to ignore empty images or non-existent file paths. Defaults to False.
       """

    def __init__(self,
                 channels,
                 data_root,
                 gt_root,
                 ann_file=None,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 color_type='color',
                 imdecode_backend='cv2',
                 nan=255,
                 downsample_factor=10,
                 pad_size=None,
                 with_seg=False,
                 GT_type=['SOD'],
                 ignore_empty=False):
        self.channels = channels
        self.mean = mean
        self.std = std
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
        self.data_root = data_root
        self.gt_root = gt_root
        self.downsample_factor = downsample_factor
        self.nan = nan
        self.with_seg = with_seg
        self.GT_type = GT_type
        self.pad_size = pad_size
        self.nc_files = self.list_nc_files(data_root, ann_file)
        self.pre_loaded_image_dic = {}
        self.pre_loaded_seg_dic = {}
        self.down_factor_perscene = {}
        ic('Starting to load images into memory...')

        for filename in tqdm(self.nc_files):

            data = {}
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            rows, cols = xarr['nersc_sar_primary'].shape

            # ----------- DOWN SCALE SETUP ON SAR CHANNELS
            rows_down, cols_down = rows, cols
            if self.downsample_factor == -1:
                downsample_factor = np.random.randint(2, 10)
            self.down_factor_perscene[filename] = downsample_factor

            if downsample_factor > 1:
                # Block average 2x2 before downsampling SAR channels
                rows_down, cols_down =  int(rows//2//(downsample_factor/2)), \
                                        int(cols//2//(downsample_factor/2))
                if 'nersc_sar_primary' in self.channels:
                    data['nersc_sar_primary'] = torch.nn.functional.avg_pool2d(torch.from_numpy(xarr['nersc_sar_primary'].values).unsqueeze(0).unsqueeze(0), 
                                                                               kernel_size = 2, stride = 2).squeeze().numpy()
                    data['nersc_sar_primary'] = torch.nn.functional.interpolate(input=torch.from_numpy(xarr['nersc_sar_primary'].values).unsqueeze(0).unsqueeze(0),
                                                                                size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()
                if 'nersc_sar_secondary' in self.channels:
                    data['nersc_sar_secondary'] = torch.nn.functional.avg_pool2d(torch.from_numpy(xarr['nersc_sar_secondary'].values).unsqueeze(0).unsqueeze(0), 
                                                                               kernel_size = 2, stride = 2).squeeze().numpy()
                    data['nersc_sar_secondary'] = torch.nn.functional.interpolate(input=torch.from_numpy(xarr['nersc_sar_secondary'].values).unsqueeze(0).unsqueeze(0),
                                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()

            # ----------- OBTAIN GRID VARIABLES
            grid_variables = self.interpolate_grid_variables(xarr, data.copy(), rows, cols, rows_down, cols_down)
            # ----------- OBTAIN ACQUISITION TIME
            scene_file_name = os.path.split(filename)[1]
            month, day = self.get_time_of_year(scene_file_name[17:32] + '_' + scene_file_name[77:80])

            # ----------- OBTAIN REMAINING VARIABLES AND NORMALIZE
            for ch in self.channels:
                if ch == 'month':
                    data[ch] = np.ones((rows_down, cols_down)) * month
                elif ch == 'day':
                    data[ch] = np.ones((rows_down, cols_down)) * day
                elif ch in ['sar_grid_latitude', 'sar_grid_longitude', 'sar_grid_incidenceangle']:
                    data[ch] = grid_variables[ch]
                else:
                    if downsample_factor > 1:
                        if ch != 'nersc_sar_primary' and ch != 'nersc_sar_secondary':
                            data[ch] = torch.nn.functional.interpolate(input=torch.from_numpy(xarr[ch].values).unsqueeze(0).unsqueeze(0),
                                                                    size=(rows_down, cols_down), mode='bilinear').numpy().squeeze()
                    else:
                        data[ch] = xarr[ch].values
                # Normalize
                data[ch] = (data[ch] - mean[ch]) / std[ch]    

            img = np.asarray(list(data.values())).transpose((1, 2, 0))
            if self.to_float32:
                img = img.astype(np.float32)
            self.pre_loaded_image_dic[filename] = img

            # ----------- OBTAIN GROUND TRUTH
            if self.with_seg:
                seg_maps = []
                for gt_type in self.GT_type:
                    gt_filename = os.path.basename(filename).replace('.nc', f'_{gt_type}.png')
                    gt_filename = os.path.join(self.gt_root, gt_filename)
                    gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
                    if downsample_factor > 1:
                        gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0)
                        gt_seg_map = torch.nn.functional.interpolate(gt_seg_map, size=(rows_down, cols_down),
                                                                     mode='nearest').numpy().squeeze()
                    seg_maps.append(gt_seg_map)
                self.pre_loaded_seg_dic[filename] = np.stack(seg_maps, axis=-1)
        ic('Finished loading images into memory...')

    def list_nc_files(self, folder_path, ann_file):
        nc_files = []
        if ann_file is not None:
            with open(ann_file, "r") as file:
                filenames = file.readlines()
            nc_files = [os.path.join(folder_path, filename.strip()) for filename in filenames]
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".nc"):
                        nc_files.append(os.path.join(root, file))
        return nc_files

    def interpolate_grid_variables(self, xarr, data, rows, cols, rows_down, cols_down):
        # ----------- INTERPOLATE GRID VARIABLES TO MATCH SAR

        grid_variables = ['sar_grid_latitude', 'sar_grid_longitude', 'sar_grid_incidenceangle']

        # Extract and reshape the initial x and y coordinates
        x = xarr['sar_grid_sample'].values
        x_l = np.unique(x)

        y = xarr['sar_grid_line'].values
        y_l = np.unique(y)

        # Define the finer grid for interpolation
        x_fine = np.linspace(0, cols - 1, cols_down)
        y_fine = np.linspace(0, rows - 1, rows_down)
        X, Y = np.meshgrid(x_fine, y_fine, indexing='xy')
        points = np.array([Y.flatten(), X.flatten()]).T

        # Loop through each variable to reshape, create interpolator, interpolate, and store results
        for var in grid_variables:
            if var in self.channels:
                values = xarr[var].values
                reshaped_values = np.reshape(values, (len(y_l), len(x_l)))
                interpolator = RegularGridInterpolator((y_l, x_l), reshaped_values, method='linear')
                interpolated_values = interpolator(points)
                data[var] = interpolated_values.reshape(X.shape)
        
        return data

    def get_time_of_year(self, file_name):
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

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        
        img_ = self.pre_loaded_image_dic[filename]
        if self.to_float32:
            img = img_.astype(np.float32)
        img = np.nan_to_num(img, nan=self.nan)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        if self.with_seg:
            seg_maps = self.pre_loaded_seg_dic[filename]
            seg_maps = np.nan_to_num(seg_maps, nan=self.nan)        # NaN values in labels
            seg_maps[np.isnan(img_[:,:,0])] = self.nan              # NaN values in image

            results['gt_seg_map'] = seg_maps
            results['seg_fields'].append('gt_seg_map')
        
        results['dws_factor'] = self.down_factor_perscene[filename]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'channels={self.channels}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'ignore_empty={self.ignore_empty})')
        return repr_str


# @TRANSFORMS.register_module()
# class LoadGTFromPNGFile(BaseTransform):
#     """Load multiple types of ground truth segmentation maps from PNG files.

#     This transform loads multiple types of ground truth segmentation maps from
#     PNG files, concatenating them into a single segmentation map with multiple channels.

#     Required Keys:
#         - img_path: Path to the NetCDF file containing the image data.

#     Modified Keys:
#         - gt_seg_map: The loaded segmentation map as a numpy array with multiple channels.
#         - seg_fields: List of segmentation fields, updated to include 'gt_seg_map'.

#     Args:
#         gt_root (str): Root directory of the ground truth segmentation maps.
#         GT_types (list[str]): List of types of ground truth to load (e.g., ['SOD', 'SIC', 'FLOE']).
#         downsample_factor (int): Factor by which to downsample the segmentation maps. Defaults to 10.
#         pad_size (tuple[int], optional): Desired size to pad the segmentation maps to. Defaults to None.
#         pad_val (float): Value to pad the segmentation maps with. Defaults to 255.
#     """

#     def __init__(self,
#                  gt_root,
#                  GT_type=['SOD'],
#                  downsample_factor=10,
#                  pad_size=None,
#                  pad_val=255):
#         self.gt_root = gt_root
#         self.GT_types = GT_type
#         self.downsample_factor = downsample_factor
#         self.pad_size = pad_size
#         self.pad_val = pad_val

#     def transform(self, results):
#         """Load multiple types of ground truth segmentation maps.

#         Args:
#             results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded segmentation map and meta information.
#         """
#         filename = results['img_path']
#         gt_maps = []
#         for GT_type in self.GT_types:
#             gt_filename = os.path.basename(filename).replace('.nc', f'_{GT_type}.png')
#             gt_filename = os.path.join(self.gt_root, gt_filename)
#             gt_seg_map = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
#             shape = gt_seg_map.shape

#             if self.downsample_factor != 1:
#                 gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0)
#                 gt_seg_map = torch.nn.functional.interpolate(
#                     gt_seg_map, size=(shape[0] // self.downsample_factor, shape[1] // self.downsample_factor), mode='nearest')
#                 gt_seg_map = gt_seg_map.squeeze(0).squeeze(0)

#                 if self.pad_size is not None:
#                     pad_height = max(0, self.pad_size[0] - gt_seg_map.shape[0])
#                     pad_width = max(0, self.pad_size[1] - gt_seg_map.shape[1])
#                     gt_seg_map = torch.nn.functional.pad(
#                         gt_seg_map, (0, pad_width, 0, pad_height), mode='constant', value=self.pad_val)
#                 gt_seg_map = gt_seg_map.numpy()

#             gt_maps.append(gt_seg_map)

#         # Concatenate the segmentation maps along the channel dimension
#         gt_seg_map = np.stack(gt_maps, axis=0)

#         results['gt_seg_map'] = gt_seg_map
#         results['seg_fields'].append('gt_seg_map')
#         return results

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'gt_root={self.gt_root}, '
#                     f'GT_types={self.GT_types}, '
#                     f'downsample_factor={self.downsample_factor}, '
#                     f'pad_size={self.pad_size}, '
#                     f'pad_val={self.pad_val})')
#         return repr_str
