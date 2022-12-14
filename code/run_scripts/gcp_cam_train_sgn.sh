
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/envs

# ================= MODIFY =========================
# current models for backbonr: VGG19_bn, VGG_att, ViT
model_class=VGG19_bn
bs=24 #24 for vgg19bn on t4, 24 for vgg19bn-bsgn on a100
ne=3 # previously ran for 4 epochs

model1="/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_EMBEDDER_epoch6.pt"
model2="/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_SHALLOW_epoch6.pt"
mtl="/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_MTL_epoch6.pt"

# eventually add embeds
embeds_path="/home/cache/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background-curr_embeddings_train.obj"

# gamified learning params
#-------------------------
save_embeds_flag=True

gamified_flag=True
backprop_level=blindfolded
# options are {none, blindfolded, full} --> none is the same as regularization term
pool_type=max # mean not yet efficiently implemented

# dataset params
#----------------
dn=cam
cd=3
scenario=cam16
ps=224
dp=/home/data/tinycam/train/train.hdf5
# ===============================================

filtration=background #none OR background; background for MISO paper
plab=inherit 
dlt=hdf5
hp=0.01
nf=False
pload=random
ploss=bce

cp=/home/cache/cam
ldp="/home/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
plp="/home/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
mp=/home/codex_analysis/codex-analysis/models/cam
desc=${model_class}"-"${dlt}"_"${pload}"_loading-"${ps}"-label_"${plab}"-"${ploss}"_loss-on_"${dn}"-"${scenario}"-filtration_"${filtration}
script=/home/codex_analysis/codex-analysis/code/train.py

# --embeds_to_load ${embeds_path} 
python ${script} --embeds_to_load ${embeds_path} --model_to_load ${model1} --model2_to_load ${model2} --mtl_to_load ${mtl} --description ${desc} --model_class ${model_class} --num_epochs ${ne} --hyperparameters ${hp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --model_path ${mp} --cache_path ${cp} --save_embeds_flag ${save_embeds_flag} --gamified_flag ${gamified_flag} --backprop_level ${backprop_level} --pool_type ${pool_type}

