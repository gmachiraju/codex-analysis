#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=20G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/y.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/y.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=yGSPatt_bg

module load python/3.6.1
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

# ================== MODIFY =====================
model=VGG_att
scenario=guilty_superpixels

# model: {VGG19, VGG_att}
# scenario:
#     {distribution_shifted_pixels
#      extreme_value_pixels
#      extreme_value_superpixels
#      guilty_superpixels
#      morphological_differences
#      fractal_morphologies
#      morphological_differences_superpixels}
# ===============================================
filtration=background # background, none
# string formatting will change after controls
ps=96    # 96, 224 for MISO-2
bs=36    # 36, 7 for MISO-2
plab=seg

# defaults
dn=controls
ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/test-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
# used to be train-"${dn}"

if [ ${plab} = "seg" ]; then
    ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-seg_labels_TEST.obj"
    echo $ldp
fi

# ============
# predictions
# ============
script1=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/predict.py
desc="${model}-stored_random_loading-${ps}-label_${plab}-bce_loss-on_MISO-1"
mc="${model}"
mchoice=manual
hp=0.01
cd=1
nf=False
dn=controls
dlt=stored
pload=random
ploss=bce

mp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/models/controls/${model}-${dlt}_${pload}_loading-${ps}-label_${plab}-${ploss}_loss-on_${dn}-${scenario}-filtration_${filtration}_full.pt"

dp="/oak/stanford/groups/paragm/gautam/syncontrols/patches/1-channel/"${scenario}"/"${ps}"-patchsize/"${filtration}"-filtered/test"
rp="/oak/stanford/groups/paragm/gautam/syncontrols/1-channel/"${scenario}"/test"
plp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/test-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"

sp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs

python ${script1} --description ${desc} --model_class ${mc} --model_choice ${mchoice}  --model_path ${mp} --hyperparameters ${hp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --reference_path ${rp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp}

# =========
# saliency
# =========
script2=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/saliency.py

pdp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/${model}-${dlt}_${pload}_loading-${ps}-label_${plab}-${ploss}_loss-on_${dn}-${scenario}-filtration_${filtration}_preddict.obj"
sr=patch
cn=${dn}"-"${scenario}"-"${ps}"-"${filtration}

python ${script2} --model_path ${mp} --model_class ${mc} --batch_size ${bs} --channel_dim ${cd} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --cache_name ${cn} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --preddict_path ${pdp} --saliency_resolution ${sr}



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