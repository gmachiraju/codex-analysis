source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/en

# ================== MODIFY =====================
model=VGG19_bn
# model: {VGG19_bn, VGG_att}

#model path:
# mp=/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_EMBEDDER_epoch1.pt
mp=/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch7.pt
# pdp=
pdp=/home/codex_analysis/codex-analysis/code/outputs/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch7_preddict.obj

#/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch2.pt
game_desc=patchcnn
arm=val
# ===============================================

scenario=cam16
dn=cam

filtration=background
ps=224
bs=24
plab=inherit

# =========
# saliency
# =========
script2=/home/codex_analysis/codex-analysis/code/saliency.py
sr=patch
bs=36

desc="${model}-stored_random_loading-${ps}-label_${plab}-bce_loss-on_cam"
cn=${desc}
# cn=${dn}"-"${scenario}"-"${ps}"-"${filtration}

mc="${model}"
mchoice=manual
cd=3
nf=False
dlt=hdf5
pload=random

dp=/home/data/tinycam/val/val.hdf5
ldp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
plp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
sp=/home/codex_analysis/codex-analysis/code/outputs

python ${script2} --model_path ${mp} --model_class ${mc} --batch_size ${bs} --channel_dim ${cd} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --cache_name ${cn} --data_path ${dp} --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --preddict_path ${pdp} --saliency_resolution ${sr}
