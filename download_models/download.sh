source /sds_wangby/group_conda_envs/init.sh
conda activate fapy310


model=model_name
log_folder=/path/to/your/logs
log_name=$model.log
mkdir -p $log_folder 

python -u /your/path/to/Download_your_tfmr.py \
   --repo_id author_name/model_name \
   --save_path /save/path/to/model \
   --token your_hf_token > ${log_folder}/$log_name 2>&1 &

