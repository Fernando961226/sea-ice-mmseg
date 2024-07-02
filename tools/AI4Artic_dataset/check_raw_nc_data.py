import h5netcdf

# Replace 'your_file.nc' with the path to your NetCDF file
nc_file = '/home/j3hsiao/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/S1A_EW_GRDM_1SDH_20190105T102726_20190105T102828_025341_02CE03_7680_icechart_cis_SGRDIEA_20190105T1023Z_pl_a.nc'

# Open the NetCDF file for reading
with h5netcdf.File(nc_file, 'r') as f:
    # Print file-level attributes
    print("Global attributes:")
    for attr_name in f.attrs:
        print(f"{attr_name}: {f.attrs[attr_name]}")
    print()

    # Print dimensions
    print("Dimensions:")
    for dim_name, dim in f.dimensions.items():
        print(f"{dim_name}: {len(dim)}")
    print()

    # Print variables and their attributes
    print("Variables:")
    for var_name, var in f.variables.items():
        print(f"Variable: {var_name}")
        print(f"Shape: {var.shape}")
        print(f"Attributes:")
        for attr_name in var.attrs:
            print(f"  {attr_name}: {var.attrs[attr_name]}")
        print()

        # Print variable values if they are not too large
        if var_name in ['sar_grid_latitude', 'sar_grid_longitude']:
            print(f"Values: {var[:]}")
            print()

# Ensure the NetCDF file is closed automatically after exiting the 'with' block

