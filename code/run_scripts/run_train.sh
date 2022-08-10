#!/bin/bash
#SBATCH --time=65:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train.out
#SBATCH --error=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/runs/train.err
#SBATCH --partition=andrewg
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gmachi@stanford.edu
#SBATCH --job-name=19_gsp-seg-bg

module load python/3.6.1
source /home/users/gmachi/anaconda3/etc/profile.d/conda.sh
conda activate /home/users/gmachi/anaconda3/envs/codex

# ================= MODIFY =========================
model_class=VGG19
scenario=guilty_superpixels

# model_class: {VGG19, VGG_att}
# scenario:
#     {distribution_shifted_pixels
#      extreme_value_pixels
#      extreme_value_superpixels
#      guilty_superpixels
#      morphological_differences
#      fractal_morphologies
#      morphological_differences_superpixels}
# ===============================================

filtration=background #none OR background; background for MISO paper
ps=96 #224 for MISO-2, 96 for MISO-1
bs=36 #7 for MISO-2, 36 for MISO-1
ne=10 #20 for MISO-2, 10 for MISO-1
plab=seg # inherit for MISO paper, can also have seg for MIL

# defaults
dn=controls
ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"

if [ ${plab} = "seg" ]; then
    ldp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-seg_labels_TRAIN.obj"
    echo $ldp
fi

# string format will be different outside of controls

script=/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/train.py
hp=0.01
cd=1
nf=False
dlt=stored
pload=random
ploss=bce
dp="/oak/stanford/groups/paragm/gautam/syncontrols/patches/1-channel/"${scenario}"/"${ps}"-patchsize/"${filtration}"-filtered/train"
plp="/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
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