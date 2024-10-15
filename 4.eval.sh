 experiment_name=Qwen2-7B_Test
 log_folder="./logs/${experiment_name}"
 result_folder="./results/${experiment_name}"
 mkdir -p $log_folder
 mkdir -p $result_folder
 log_name=$(date +"%m-%d_%H-%M").log

 accelerate launch /path/to/src/eval/eval_qwen2.py \
     --model_path=/path/to/ckpts/Qwen2-7B_train/checkpoint-epochs-steps/tfmr \
     --input_path=/path/to/Qwen2-7B_Prodata/test.json \
     --output_path=${result_folder}/model_ans_test.jsonl \
     --score_path=${result_folder}/score.json \
     --num_return=1 \
     --batch_size=4 > ${log_folder}/$log_name 2>&1 &\


#########################################################
 experiment_name=Apollo-MoE-0.5B_Test
 log_folder="./logs/${experiment_name}"
 result_folder="./results/${experiment_name}"
 mkdir -p $log_folder
 mkdir -p $result_folder
 log_name=$(date +"%m-%d_%H-%M").log

 model_dir=/path/to/ckpts/Apollo-MoE-0.5B_train/checkpoint-epochs-steps/tfmr
 train_dir=/path/to/src/variants/upcycling/language_family
 cp ${model_dir}/configuration_upcycling_qwen2_moe.py ${train_dir}/configuration_upcycling_qwen2_moe.py
 cp ${model_dir}/modeling_upcycling_qwen2_moe.py ${train_dir}/modeling_upcycling_qwen2_moe.py

 accelerate launch /path/to/src/variants/upcycling/language_family/eval_qwen2.py \
     --model_path=${model_dir} \
     --input_path=/path/to/metadata/test/test.json \
     --output_path=${result_folder}/model_ans_test.jsonl \
     --score_path=${result_folder}/score.json \
     --num_return=1 \
     --batch_size=4 > ${log_folder}/$log_name 2>&1 &\
