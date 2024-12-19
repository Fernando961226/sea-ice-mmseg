set -e
# deactivate
module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"
echo "Creating new virtualenv"
virtualenv ~/env_mmselfsup
source ~/env_mmselfsup/bin/activate

echo "Activating virtual env"

pip install jupyterlab
pip install ipywidgets
pip install xarray
pip install h5netcdf

pip install sklearn
pip install matplotlib
pip install numpy
pip install icecream
pip install termcolor

pip install tqdm
pip install joblib
pip install wandb==0.18.0

pip install torch torchvision torchmetrics torch-summary

pip install ftfy
pip install regex
pip install ninja psutil

pip install mmengine>=0.8.3
pip install mmcv
# If there is any conflict installing mmcv, 
# install the package from scratch:
# https://mmcv.readthedocs.io/en/latest/get_started/build.html (branch 2.x)

cd ../../../sea-ice-mmselfsup/
pip install -U openmim && mim install -e .

cd ../sea-ice-mmseg/
pip install -v -e .

# _dir=$(pwd)
# cd $_dir