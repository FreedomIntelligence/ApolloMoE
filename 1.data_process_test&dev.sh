# Major and Minor Test Data for Apollo2 trained with Qwen2
python ./src/prepare/data_process_test_qwen2.py \
 --data_path ./metadata/test.json \
 --few_shot 3 \
 --save_path ./data/Apollo2-7B_Prodata/test.json

python ./src/prepare/data_process_test_qwen2.py \
 --data_path ./metadata/dev.json \
 --few_shot 3 \
 --save_path ./data/Apollo2-7B_Prodata/dev.json
 
python ./src/prepare/data_process_test_qwen2.py \
 --data_path ./metadata/minor_test.json \
 --few_shot 3 \
 --save_path ./data/Apollo2-7B_Prodata/test.json

