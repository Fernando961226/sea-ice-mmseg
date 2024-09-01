set -e
# deactivate
module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"
echo "Creating new virtualenv"
virtualenv ~/env_mmsegmentation
source ~/env_mmsegmentation/bin/activate

cd ../../

pip install tqdm
pip install sklearn
pip install jupyterlab
pip install ipywidgets
pip install icecream
pip install wandb
pip install matplotlib
pip install numpy
pip install xarray
pip install h5netcdf
pip install ftfy
pip install regex
pip install joblib
pip install torch torchvision torchmetrics torch-summary

# pip install mim
pip install mmengine>=0.8.3
pip install mmcv
# If there is any conflict installing mmcv, 
# install the package from scratch:
# https://mmcv.readthedocs.io/en/latest/get_started/build.html
pip install -v -e .


# # mim installation
# pip install -U openmim
# mim install mmengine


# # build mmcv from source-- NEED TO HAVE GPU FOR THIS
# git clone -b "2.x" --single-branch https://github.com/open-mmlab/mmcv.git
# cd ../mmcv
# MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e .

# # install mmcv from pip
# # MMCV_WITH_OPS=1 FORCE_CUDA=1  mim install mmcv --no-deps

# # install mmde
# cd $mmwhale_dir
# pip install -v -e .

# # pip uninstall opencv-python
# # pip uninstall opencv-python-headless
# # pip install opencv-python-headless
