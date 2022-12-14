source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/envs

# ================== MODIFY =====================
model=VGG19_bn
# model: {VGG19_bn, VGG_att}

#model path:
# mp=/home/codex_analysis/codex-analysis/models/cam/gamify-uncertainty-backpropblindfolded-max_pooling-VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_EMBEDDER_epoch1.pt
mp=/home/codex_analysis/codex-analysis/models/cam/VGG19_bn-hdf5_random_loading-224-label_inherit-bce_loss-on_cam-cam16-filtration_background_epoch7.pt

game_desc=patchcnn
arm=val
# ===============================================

scenario=cam16
dn=cam

filtration=background
ps=224
bs=24
plab=inherit

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

dp=/home/data/tinycam/val/val.hdf5
# rp=None
ldp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"
plp="/home/codex_analysis/codex-analysis/code/outputs/"${arm}"-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-patchlist.obj"
sp=/home/codex_analysis/codex-analysis/code/outputs

python ${script1} --description ${desc} --model_class ${mc} --model_choice ${mchoice} --model_path ${mp} --batch_size ${bs} --channel_dim ${cd} --normalize_flag ${nf} --dataset_name ${dn} --dataloader_type ${dlt} --patch_size ${ps} --patch_loading ${pload} --patch_labeling ${plab} --patch_loss ${ploss} --data_path ${dp}  --patchlist_path ${plp} --labeldict_path ${ldp} --save_path ${sp} --game_description ${game_desc}
