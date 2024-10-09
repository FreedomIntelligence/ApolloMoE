cd /path/to/src/proprcess/moe_initilization

#Initilization from Base model
model=Model_Name
experiment_name=convert_${model}
log_folder="./logs/${experiment_name}"
result_folder="./results/${experiment_name}"
mkdir -p $log_folder
mkdir -p $result_folder
log_name=$(date +"%m-%d_%H-%M").log

#--share_flag     True for share expert
#--language_gate        True for language-specific routing
#--test  True for test
python -u /path/to/src/upcycling_qwen_new/convert_qwen_moe.py \
    --language_gate \
    --model_path=/path/to/Qwen2-7B \
    --output_path=/path/to/models/${model} \
    > ${log_folder}/$log_name 2>&1 &
