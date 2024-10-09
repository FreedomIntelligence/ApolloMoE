
 process_port=29503
 experiment_name=Qwen2-7B_train
 model_dir=/path/to/Qwen2-7B
 train_data_file=/path/to/Qwen2-7B_Prodata/Mixtrain_Prodata
 dev_data_file=/path/to/Qwen2-7B_Prodata/dev.json
 output_dir=/path/to/ckpts
 log_folder=/path/to/logs/${experiment_name}
 mkdir -p $log_folder
 log_name=$(date +"%m-%d_%H-%M").log

 accelerate launch \
     --config_file /path/to/src/sft/training_config/zero.yaml \
     --num_processes 8 \
     --num_machines 1 \
     --main_process_port ${process_port} \
     --num_cpu_threads_per_process 2 \
     --deepspeed_multinode_launcher standard /path/to/src/sft/train_qwen2_resume_val.py \
     --model_path ${model_dir} \
     --experiment_name ${experiment_name} \
     --gradient_accumulation_steps 2 \
     --train_data_dir ${train_data_file} \
     --dev_data_dir ${dev_data_file} \
     --output_dir ${output_dir} \
     --log_dir ./wandb_logs \
     --n_epochs 1 \
     --train_bsz_per_gpu 2 \
     --eval_bsz_per_gpu 2 \
     --learning_rate 1e-5 \
     --eval_step -1 \
     --save_step -1 \
     --warmup_rates 0.03 \
     --max_ckpts 3 \
     --gradient_checkpointing  > ${log_folder}/$log_name 2>&1 &

#########################################################################################

 process_port=29502
 experiment_name=Apollo-MoE-0.5B_train
 model_dir=/path/to/Apollo-MoE-0.5B-Base
 train_data_file=/path/to/Post-MoE-Qwen2-0.5B_Prodata/Mixtrain_Prodata
 dev_data_file=/path/to/Post-MoE-Qwen2-0.5B_Prodata/dev.json
 output_dir=/path/to/ckpts
 log_folder=/path/to/logs/${experiment_name}
 mkdir -p $log_folder
 log_name=$(date +"%m-%d_%H-%M").log

 train_dir=/path/to/src/variants/upcycling/language_family
 cp ${model_dir}/configuration_upcycling_qwen2_moe.py ${train_dir}/configuration_upcycling_qwen2_moe.py
 cp ${model_dir}/modeling_upcycling_qwen2_moe.py ${train_dir}/modeling_upcycling_qwen2_moe.py

 accelerate launch \
     --config_file /path/to/sft/training_config/zero.yaml \
     --num_processes 8 \
     --num_machines 1 \
     --main_process_port ${process_port} \
     --num_cpu_threads_per_process 2 \
     --deepspeed_multinode_launcher standard /path/to/src/variants/upcycling/language_family/train_qwen2_resume_val_upcycling.py \
     --model_path ${model_dir} \
     --experiment_name ${experiment_name} \
     --gradient_accumulation_steps 2 \
     --train_data_dir ${train_data_file} \
     --dev_data_dir ${dev_data_file} \
     --output_dir ${output_dir} \
     --log_dir ./wandb_logs \
     --n_epochs 1 \
     --train_bsz_per_gpu 2 \
     --eval_bsz_per_gpu 2 \
     --learning_rate 1e-4 \
     --eval_step -1 \
     --save_step -1 \
     --warmup_rates 0.03 \
     --max_ckpts 3 \
     --language_gate \
     --gradient_checkpointing  > ${log_folder}/$log_name 2>&1 &

