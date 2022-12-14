
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/envs

# ================= MODIFY =========================
# current models for backbone: VGG19_bn, VGG_att, ViT
model_class=ResNet18
ne=5 # usually like 1-10
overfit_flag=False

# model1="/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch7.pt"
# model1="/home/codex_analysis/codex-analysis/models/cam/ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background_epoch4.pt"
# model1="/home/codex_analysis/codex-analysis/models/cam/CARTA-ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background.sd"

# model1="/home/codex_analysis/codex-analysis/models/cam/ResNet18-hdf5_triplets_random_loading-224-label_selfsup-custom_loss-on_cam-cam16-filtration_background.sd"

selfsup_mode=mix        # mix for regular self-sup  // sextuplet for carta
coaxial_flag=False      # False for regular self-sup // True for carta
bs=20                   # 15-20 for regular self-sup // 5 for carta


# gamified learning params
#-------------------------
save_embeds_flag=False
gamified_flag=False
backprop_level=none
# options are {none, blindfolded, full} --> none is the same as regularization term
pool_type=max # mean not yet efficiently implemented

# self-sup params
#----------------
selfsup_flag=True
trip0_path="/home/data/tinycam/train/triplets0_list.obj"
trip1_path="/home/data/tinycam/train/triplets1_list.obj"

# dataset params
#----------------
dn=cam
cd=3
scenario=cam16
ps=224
dp=/home/data/tinycam/train/train.hdf5
# ===============================================

filtration=background #none OR background; background for MISO paper
plab=selfsup 
dlt=hdf5_triplets

hp=0.01
nf=False
pload=random
ploss=custom

cp=/home/cache/cam
ldp="/home/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
plp="/home/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
mp=/home/codex_analysis/codex-analysis/models/cam
desc=${model_class}"-"${dlt}"_"${pload}"_loading-"${ps}"-label_"${plab}"-"${ploss}"_loss-on_"${dn}"-"${scenario}"-filtration_"${filtration}
script=/home/codex_analysis/codex-analysis/code/train.py

# --model_to_load ${model1}
python ${script} --coaxial_flag ${coaxial_flag} --overfit_flag ${overfit_flag}  --selfsup_flag ${selfsup_flag} --trip0_path ${trip0_path} --trip1_path ${trip1_path} --selfsup_mode ${selfsup_mode}  --description ${desc} --model_class ${model_class} --num_epochs ${ne} --hyperparameters ${hp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --model_path ${mp} --cache_path ${cp} --save_embeds_flag ${save_embeds_flag} --gamified_flag ${gamified_flag} --backprop_level ${backprop_level} --pool_type ${pool_type}
