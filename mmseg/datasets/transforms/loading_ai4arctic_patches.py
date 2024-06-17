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
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
import numpy as np
from AI4ArcticSeaIceChallenge.convert_raw_icechart import convert_polygon_icechart
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

        ## location and incidence angle
            -   sar_grid_latitude
            -   sar_grid_longitude
            -   sar_grid_incidenceangle

        #TODO: list environmental variables

        ## acquisition time
            -   month
            -   day

        ## info
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
           data_root (str): Root directory of the patch files.
           gt_root (str): Root directory of the ground truth segmentation maps.
           ann_file (str, optional): Path to the annotation file listing NetCDF files to load.
           mean (list[float]): Mean values for normalization of each channel. Defaults to values provided.
           std (list[float]): Standard deviation values for normalization of each channel. Defaults to values provided.
           to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. Defaults to True.
           color_type (str): The color type for image loading. Defaults to 'color'.
           imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
           nan (float): Value to replace NaNs in the image. Defaults to 255.
           
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
        # self.nc_files = self.list_nc_files(ann_file)

        self.img_files = []
        self.ann_files = []
        
        for d in self.downsample_factor:
            if ann_file is not None:
                with open(ann_file, "r") as file:
                    filenames = file.readlines()
                    for file in filenames:
                        image = os.path.splitext(file)[0]
                        self.img_files.extend(glob.glob(os.path.join(self.data_root, 'down_scale_%dX/%s/*.pkl'%(d, image))))
                        self.ann_files.extend(glob.glob(os.path.join(self.gt_root  , 'down_scale_%dX/%s/*.pkl'%(d, image))))
            else:
                self.img_files.extend(glob.glob(os.path.join(self.data_root, 'down_scale_%dX/**/*.pkl'%(d))))

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        img = self.pre_loaded_image_dic[filename]

        
        if self.to_float32:
            img = img.astype(np.float32)
        img = np.nan_to_num(img, nan=self.nan)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        if self.with_seg:
            results['gt_seg_map'] = self.pre_loaded_seg_dic[filename]
            results['seg_fields'].append('gt_seg_map')
        return results
