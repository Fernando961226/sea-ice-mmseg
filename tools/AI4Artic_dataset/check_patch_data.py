"""
File use to check the output of patch_dataset_creator_raw.py and patch_dataset_creator
"""


#%%
import joblib
import numpy as np

# Specify the path to the saved .pkl file
file_path = '/home/j3hsiao/scratch/dataset/ai4arctic/07-02_15-27/down_scale_10X/S1A_EW_GRDM_1SDH_20210914T080148_20210914T080253_039675_04B0EB_9542_icechart_dmi_202109140800_North_RIC/00000.pkl'

# Load the dictionary from the .pkl file
data_patch = joblib.load(file_path)

# Check if 'sar_pos_spherical' is in the loaded dictionary
if 'sar_pos_spherical' in data_patch:
    sar_pos_spherical = data_patch['sar_pos_spherical']
    
    # Print the shape of the array
    print("Shape of 'sar_pos_spherical':", np.shape(sar_pos_spherical))
    
    # Print the values of the array
    print("Values of 'sar_pos_spherical':")
    print(sar_pos_spherical)
else:
    print("'sar_pos_spherical' not found in the loaded data patch.")

# %%

keys = ['nersc_sar_primary',
        'nersc_sar_secondary',
        'sar_nan_mask',
        'sar_grid_latitude',
        'sar_grid_longitude',
        'sar_grid_incidenceangle',
        'distance_map',
        'btemp_6_9h',
        'btemp_6_9v',
        'btemp_7_3h',
        'btemp_7_3v',
        'btemp_10_7h',
        'btemp_10_7v',
        'btemp_18_7h',
        'btemp_18_7v',
        'btemp_23_8h',
        'btemp_23_8v',
        'btemp_36_5h',
        'btemp_36_5v',
        'btemp_89_0h',
        'btemp_89_0v',
        'u10m_rotated',
        'v10m_rotated',
        't2m',
        'skt',
        'tcwv',
        'tclw',
        'SIC',
        'SOD',
        'FLOE']

for key in keys:
    plt.figure()
    plt.imshow(data_patch[key])
    plt.title(key)
    plt.show()
# %%
data = joblib.load('/home/fernando/Documents/Graduate_Studies/Python/sea-ice-mmpretrain/out.pkl')

for key in keys:
    print(data[key].shape)
# %%
