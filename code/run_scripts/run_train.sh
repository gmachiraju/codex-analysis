#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train_VGG_att.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train_VGG_att.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=VGG_att_fm

module load python/3.6.1
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

# ================= MODIFY =========================
model_class=VGG_att
scenario=fractal_morphologies

# model_class: {VGG19, VGG_att}
# scenario:
#     {distribution_shifted_pixels
#      extreme_value_pixels
#      extreme_value_superpixels
#      guilty_superpixels
#      morphological_differences
#      fractal_morphologies}
# ===============================================

filtration=background
# string format will be different outside of controls


script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/train.py
ne=10
hp=0.01
bs=36
cd=1
nf=False
dn=controls
dlt=stored
ps=96
pload=random
plab=inherit
ploss=bce
dp="/oak/stanford/groups/paragm/gautam/syncontrols/patches/1-channel/"${scenario}"/"${ps}"-patchsize/"${filtration}"-filtered/train"
plp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
mp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/models/controls
desc=${model_class}"-"${dlt}"_"${pload}"_loading-"${ps}"-label_"${plab}"-"${ploss}"_loss-on_"${dn}"-"${scenario}"-filtration_"${filtration}


python ${script} --description ${desc} --model_class ${model_class} --num_epochs ${ne} --hyperparameters ${hp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --model_path ${mp} 




# dp=/oak/stanford/groups/paragm/gautam/syncontrols/1-channel/extreme_value_pixels/train # only for otf

# python ${script} --experiment_desc 25EPOCHuncertainty_fullImgGraph --model_class ModVGG19_proxypred_bn --num_epochs 20 --controls_testing False --batch_size 36 --alpha 0.01
# Please choose a model: 
# 		ModVGG19
# 		ModVGG19_bn
# 		ModVGG19_proxypred
# 		ModVGG19_proxypred_bn
# 		TLCSB_IN_mean
# 		TLCSB_IN_max 
# 		TLCSB_PC_mean
# 		TLCSB_PC_max 
# Or choose to pretrain VGG19 with model: 
# 		pretrainVGG19

# can try ibiis or andrewg for resources