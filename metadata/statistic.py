import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm  # 导入tqdm

model_path='/sds_wangby/models/Qwen2-1.5B'
tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')

# 指定输出文件的路径
output_file_path = '/path/to/metadata/major_languages/train/sft_token_counts.json'

# 定义要遍历的文件夹路径
folder_path = '/path/to/metadata/major_languages/train/'

if os.path.exists(output_file_path):
    with open(output_file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    final_total_tokens=0
    for k,v in data.items():
        final_total_tokens+=int(v)
else:
    #所有token数
    final_total_tokens=0
    # 用于存储每个文件的token计数
    token_counts = {}

    # 获取文件夹中所有json文件的列表
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

    # 遍历文件夹中的每个文件
    for file_name in tqdm(json_files, desc="Processing JSON files"):  # 添加进度条
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            total_tokens = 0
            # 遍历文件中的每个item
            for item in tqdm(data, desc=f"Tokenizing items in {file_name}", leave=False):  # 对每个文件的item添加进度条
                # 遍历item的每个subitem
                for subitem in item:  # 为subitems也添加进度条
                    # 使用tokenizer处理subitem
                    tokens = tokenizer.tokenize(subitem, add_special_tokens = False)
                    # 累加token数量
                    total_tokens += len(tokens)
            # 存储该文件的总token数
            token_counts[file_name] = total_tokens
            final_total_tokens+=total_tokens

    # 将token_counts字典保存到文件中
    with open(output_file_path, 'w') as output_file:
        json.dump(token_counts, output_file, indent=4)
    print(f'Token counts have been saved to {output_file_path}.')
print(f'Total tokens:{final_total_tokens/1e+9}B')

