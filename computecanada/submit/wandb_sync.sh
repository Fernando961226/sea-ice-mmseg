set -e

mapfile -t array < <(find /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg/work_dirs -type d -name "vis_data")

module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"
source ~/env_mmsegmentation/bin/activate

for i in "${!array[@]}"; do

   if [[ "${array[i]}" != *"mae_ai4arctic_ds2_pt_80_ft_20"* ]]; then 
      cd ${array[i]}
      echo "wandb sync --sync-all " ${array[i]}
      # wandb sync --sync-all ${array[i]}
      wandb sync --sync-all wandb/
      sleep 1
   else
      echo ${array[i]}
   fi
done

deactivate