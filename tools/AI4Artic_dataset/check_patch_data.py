"""
File use to check the output of patch_dataset_creator_raw.py and patch_dataset_creator
"""


#%%
import joblib
import matplotlib.pyplot as plt


# Specify the path to the saved .pkl file
file_path = '/home/fernando/scratch/train/S1A_EW_GRDM_1SDH_20190210T120052_20190210T120156_025867_02E111_12DF_icechart_dmi_201902101200_Qaanaaq_RIC_down_scale_4X/00000.pkl'

# Load the dictionary from the .pkl file
data_patch = joblib.load(file_path)

# Now 'loaded_data_patch' contains the deserialized dictionary
print(data_patch)
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
