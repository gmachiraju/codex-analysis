#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=30G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/cam.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/cam.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=train-cam_process

module load python/3.6.1
module load biology
module load py-openslide-python/1.1.1_py36
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/cam_process.py

arm=train
df=False

td="/oak/stanford/groups/paragm/gautam/cam/cam16/"${arm}
xd="/oak/stanford/groups/paragm/gautam/cam/cam_scp/lesion_annotations_"${arm}
sd="/oak/stanford/groups/paragm/gautam/cam/cam16-extract/"${arm}
sdm="/oak/stanford/groups/paragm/gautam/cam/cam16-extract/masks/"${arm}

python ${script} --arm ${arm} --tifdir ${td} --xmldir ${xd} --savedir ${sd} --savedirmasks ${sdm} --debugflag ${df}
 

