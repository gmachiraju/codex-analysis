#!/bin/bash
#SBATCH --time=0:50:00
#SBATCH --mem=2G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/cam.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/cam.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=013-cam-download

# module load python/3.6.1
# module load biology
# module load py-openslide-python/1.1.1_py36
# conda activate /home/users/gmachi/anaconda3/envs/codex

module load devel
module load python/3.9.0
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/gamify

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/cam_download.py

fn=patient_013.zip
sd=/oak/stanford/groups/paragm/gautam/cam/cam16
wc=cam17

python ${script} --filename ${fn} --savedir ${sd} --whichcam ${wc}
 