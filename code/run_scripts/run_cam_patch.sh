#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/patch.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/patch.err
#SBATCH --partition=andrewg
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=cam-val-patch

# module load python/3.6.1
# module load biology
# module load py-openslide-python/1.1.1_py36
# module load gcc/9.1.0 
# module load openblas/0.3.20 
# source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
# conda activate /home/users/gmachi/anaconda3/envs/codex

# module load devel
# module load python/3.9.0
# source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
# conda activate /home/users/gmachi/anaconda3/envs/gamify

module load python/3.6.1
module load biology
module load py-openslide-python/1.1.1_py36
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/patho

# =========== MODIFY ===================
cache_name=cam16
study_arm=val
augment_level=none # used to be none
# study_arm: {train, test, both}
ld=/oak/stanford/groups/paragm/gautam/cam/cam17-scp/val_labels.obj
# can be none if filename tells you label^
dataset_name=cam
data_dir="/oak/stanford/groups/paragm/gautam/"${dataset_name}"/"${cache_name}"/"${study_arm}
bgrf=True # is background removed? Yes for validation
# ========================================

ft=background #none #background (only have this for MISO paper)
HW=224 #224 for MISO-2 #96 for MISO-1
of=True
ogf=True

# not implemented for now
rf=False # resize_flag for saving
rHW=224 # resize dimension for saving
# need to change string formatting below for non-controls work

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/preprocess.py

prepatch_flag=True
save_dir="/oak/stanford/groups/paragm/gautam/"${dataset_name}"/"${cache_name}"-patches/"${study_arm}
cache_dir=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code
vf=True
cgtf=True
hdf5_flag=True

python ${script} --dataset_name ${dataset_name} --data_dir ${data_dir} --study_arm ${study_arm} --cache_name ${cache_name} --HW ${HW} --resize_flag ${rf} --resize_HW ${rHW} --augment_level ${augment_level} --filtration_type ${ft} --prepatch_flag ${prepatch_flag} --overwrite_flag ${of} --save_dir ${save_dir} --cache_dir ${cache_dir} --verbosity_flag ${vf} --control_groundtruth_flag ${cgtf} --overwrite_gt_flag ${ogf} --hdf5_flag ${hdf5_flag} --label_dict ${ld} --bg_remove_flag ${bgrf}

# python /home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/preprocess.py /home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_u54/primary /home/groups/plevriti/gautam/codex_analysis/codex-analysis/patches/
