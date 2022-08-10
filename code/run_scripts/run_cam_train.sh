#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=vgg_att-cam
 
module load python/3.9.0
module load math
module –ignore_cache load py-pytorch 1.11.0_py39

source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex


# ================= MODIFY =========================
model_class=VGG_att
bs=7 #7 for vgg_att, 10 for vgg19
# current models: VGG19_bn, VGG_att, ViT
# ===============================================

cd=3
scenario=cam16
filtration=background #none OR background; background for MISO paper
ps=224 
ne=20 
plab=inherit 
dlt=hdf5
dn=cam
hp=0.01
nf=False
pload=random
ploss=bce

dp="/oak/stanford/groups/paragm/gautam/"${dn}"/"${scenario}"-patches/train/train.hdf5"

ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"

plp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"

mp=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/models/cam

desc=${model_class}"-"${dlt}"_"${pload}"_loading-"${ps}"-label_"${plab}"-"${ploss}"_loss-on_"${dn}"-"${scenario}"-filtration_"${filtration}

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/train.py

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