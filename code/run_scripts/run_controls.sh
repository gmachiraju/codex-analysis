#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=20G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/controls.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/controls.err
#SBATCH --partition=ibiis
#SBATCH --gres gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=controls

module load python/3.6.1
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

# ============== MODIFY =========================
partition=distribution_shifted_pixels

# partition:
#     {distribution_shifted_pixels
#      extreme_value_pixels
#      extreme_value_superpixels
#      guilty_superpixels
#      morphological_differences
#      fractal_morphologies}
# ===============================================


script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/valid_controls.py
dp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/bitmaps
bf=False
mf=True
msp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/thresh_dict.txt
mof=True
pof=True
cd=1
split=both
sp=/oak/stanford/groups/paragm/gautam/syncontrols
mdf=False

python ${script} --data_path ${dp} --bin_flag ${bf} --manualbin_flag ${mf} --manualbin_settings_path ${msp} --mask_overwrite_flag ${mof} --partition_overwrite_flag ${pof} --channel_dim ${cd} --split ${split} --partition ${partition} --save_path ${sp} --mask_debug_flag ${mdf}
 
# could also do angrewg/ibiis
# extreme_value_pixels, distribution_shifted_pixels, morphological_differences, guilty_superpixels, extreme_value_superpixels, fractal_morphologies