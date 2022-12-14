source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/envs

# ================== MODIFY =====================
model=VGG19_bn
# model: {VGG19_bn, VGG_att}

# model path + (anticipated) predictions dict path:
# backbone
# mp=/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch9.pt
# pdp=/home/codex_analysis/codex-analysis/code/outputs/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch9_preddict.obj

# SGN: EMBEDDER
mp=/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_EMBEDDER_epoch9.pt
pdp=/home/codex_analysis/codex-analysis/code/outputs/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_EMBEDDER_epoch9_preddict.obj

game_desc=patchcnn
arm=val
dp=/home/data/tinycam/val/val.hdf5
# ===============================================

scenario=cam16
dn=cam
filtration=background
ps=224
bs=24
plab=inherit
# rp=None
ldp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
plp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
sp=/home/codex_analysis/codex-analysis/code/outputs

# ============
# predictions
# ============
script1=/home/codex_analysis/codex-analysis/code/predict.py
desc="${model}-stored_random_loading-${ps}-label_${plab}-bce_loss-on_cam"
mc="${model}"
mchoice=manual
cd=3
nf=False
dlt=hdf5
pload=random
ploss=bce

python ${script1} --description ${desc} --model_class ${mc} --model_choice ${mchoice} --model_path ${mp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp}  --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --game_description ${game_desc}

# =========
# saliency
# =========
script2=/home/codex_analysis/codex-analysis/code/saliency.py
sr=patch
bs=36

desc="${model}-stored_random_loading-${ps}-label_${plab}-bce_loss-on_cam"
cn=${desc}
# cn=${dn}"-"${scenario}"-"${ps}"-"${filtration}

python ${script2} --model_path ${mp} --model_class ${mc} --batch_size ${bs} --channel_dim ${cd} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --cache_name ${cn} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --preddict_path ${pdp} --saliency_resolution ${sr}
