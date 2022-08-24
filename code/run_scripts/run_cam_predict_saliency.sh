#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=20G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/y.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/y.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=yVGG19_bn-cam

module purge
module load devel
module load python/3.9.0
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/gamify

# ================== MODIFY =====================
model=VGG_att
# model: {VGG19_bn, VGG_att}

#model path:
mp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/models/cam/VGG_att-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch6.pt
game_desc=patchcnn
scenario=cam16
dn=cam
arm=val
# ===============================================

filtration=background
ps=224
bs=20
plab=inherit

# ============
# predictions
# ============
script1=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/predict.py
desc="${model}-stored_random_loading-${ps}-label_${plab}-bce_loss-on_cam"
mc="${model}"
mchoice=manual
cd=3
nf=False
dlt=hdf5
pload=random
ploss=bce

dp="/oak/stanford/groups/paragm/gautam/"${dn}"/"${scenario}"-patches/"${arm}"/"${arm}".hdf5"

rp="/oak/stanford/groups/paragm/gautam/"${dn}"/"${scenario}"/"${arm}""

ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"

plp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"

sp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs

python ${script1} --description ${desc} --model_class ${mc} --model_choice ${mchoice} --model_path ${mp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --reference_path ${rp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --game_description ${game_desc}

# =========
# saliency
# =========
# script2=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/saliency.py

# pdp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/${model}-${dlt}_${pload}_loading-${ps}-label_${plab}-${ploss}_loss-on_${dn}-${scenario}-filtration_${filtration}_preddict.obj"
# sr=patch
# cn=${dn}"-"${scenario}"-"${ps}"-"${filtration}

# python ${script2} --model_path ${mp} --model_class ${mc} --batch_size ${bs} --channel_dim ${cd} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --cache_name ${cn} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --preddict_path ${pdp} --saliency_resolution ${sr}



# mc="${model}"
# bs=36
# cd=1
# dn=controls
# dlt=stored
# ps=96
# mp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/models/controls/${model}-${dlt}_${pload}_loading-${ps}-label_${plab}-${ploss}_loss-on_${dn}-${scenario}-filtration_${filtration}_full.pt"
# dp=/oak/stanford/groups/paragm/gautam/syncontrols/patches/1-channel/extreme_value_pixels/test
# plp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/test_patchlist_controls_96_extreme_value_pixels.obj
# sp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs
# ldp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/test_labeldict_controls_96_extreme_value_pixels.obj
# pdp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/${model}-controls-96-random-inherit_full_preddict.obj"