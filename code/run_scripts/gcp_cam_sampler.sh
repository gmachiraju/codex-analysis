source /opt/conda/etc/profile.d/conda.sh
conda activate /home/codex_analysis/codex-analysis/envs

#-------
sn=3 # triplet sampling
dm="across" #used to be "same"
#-------

dn=cam
cd=3
scenario=cam16
ps=224
filtration=background #none OR background; background for MISO paper

dp=/home/data/tinycam/train/train.hdf5
ldp="/home/codex_analysis/codex-analysis/code/outputs/train-"${dn}"-"${scenario}"-"${ps}"-"${filtration}"-labeldict.obj"

script=/home/codex_analysis/codex-analysis/code/sampler.py

python ${script} --data_path ${dp} --sampling_number ${sn} --labeldict_path ${ldp} --dataset_name ${dn} --distant_mode ${dm}