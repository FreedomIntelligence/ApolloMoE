
experiment_name=Qwen2-7B_Prodata
log_folder=./logs/${experiment_name}
mkdir -p $log_folder
mkdir -p ./data/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

python /path/to/src/prepare/data_process_train_prodata_qwen2.py \
    --data_path /path/to/metadata/train/train_data.json \
    --model_path /path/to/Qwen2-7B \
    --wandb_log ./wandb_logs \
    --max_seq_len 4096 \
    --experiment_name ${experiment_name} \
    --save_path ./data/${experiment_name}/Mixtrain_Prodata > ${log_folder}/$log_name 2>&1 &


experiment_name=Apollo-MoE-0.5B_Prodata
log_folder=./logs/${experiment_name}
mkdir -p $log_folder
mkdir -p ./data/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

python /path/to/src/variants/upcycling/language_family/data_process_train_prodata_qwen_language_family.py \
    --data_path /path/to/metadata/train/train_data.json \
    --model_path /path/to/Qwen2-0.5B \
    --wandb_log ./wandb_logs \
    --max_seq_len 4096 \
    --experiment_name ${experiment_name} \
    --save_path ./data/${experiment_name}/Mixtrain_Prodata > ${log_folder}/$log_name 2>&1 &