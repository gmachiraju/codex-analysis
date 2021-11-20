#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/patch.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/patch.err
#SBATCH --partition=andrewg
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=evp-test-patch

module load python/3.6.1
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

# =========== MODIFY ===================
cache_name=extreme_value_pixels
study_arm=test  

# study_arm: {train, test, both}
# cache_name:
#     {distribution_shifted_pixels
#      extreme_value_pixels
#      extreme_value_superpixels
#      guilty_superpixels
#      morphological_differences
#      fractal_morphologies}
# ========================================
ft=background
# need to change string formatting below for non-controls work

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/preprocess.py
dataset_name=controls
data_dir="/oak/stanford/groups/paragm/gautam/syncontrols/1-channel/"${cache_name}"/"${study_arm}
HW=96
augment_level=none
prepatch_flag=True
of=True
save_dir="/oak/stanford/groups/paragm/gautam/syncontrols/patches/1-channel/"${cache_name}"/"${HW}"-patchsize/"${ft}"-filtered/"${study_arm}
cache_dir=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code
vf=True
cgtf=True
ogf=True
# we specify complete save_dir since this can be used for any data generally.

python ${script} --dataset_name ${dataset_name} --data_dir ${data_dir} --study_arm ${study_arm} --cache_name ${cache_name} --HW ${HW} --augment_level ${augment_level} --filtration_type ${ft} --prepatch_flag ${prepatch_flag} --overwrite_flag ${of} --save_dir ${save_dir} --cache_dir ${cache_dir} --verbosity_flag ${vf} --control_groundtruth_flag ${cgtf} --overwrite_gt_flag ${ogf}

# python /home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/preprocess.py /home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_u54/primary /home/groups/plevriti/gautam/codex_analysis/codex-analysis/patches/
