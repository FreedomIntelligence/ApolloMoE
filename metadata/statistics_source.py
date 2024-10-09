    
import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm  # 导入tqdm
import math 
from collections import defaultdict


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def get_file_size(file_path):
    try:
        file_size = os.path.getsize(file_path)
        return convert_size(file_size)
    except OSError as e:
        print(f"Error: {e}")
        return None
    
def get_file_size_MB(file_path):
    file_size=os.path.getsize(file_path)
    MB=math.pow(1024,2)
    return round(file_size/MB,2)

def tokens_count(file_path,tokenizer):
    with open(file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        total_tokens=0
        for item in data:
            for subitem in item:
                tokens=tokenizer.tokenize(subitem,add_special_tokens=False)
                total_tokens+=len(tokens)
    return round(total_tokens/1e+6,2)


languages=['_ar','_en','_es','_de','_it','_ja','_ko','_fr','_pt','_ru','_zh','_hi']
sources=['code','math','general','Book','Exam','Web','Wiki','Guideline','Paper','Patient','Web']

def statistics_token_size(folder_path,output_tokens_path,output_size_path):
    
    model_path='/sds_wangby/models/Qwen2-0.5B'
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')

    final_file_size={}
    file_size=defaultdict(dict)

    final_total_tokens={}
    token_counts =defaultdict(dict)
    
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

    for file_name in tqdm(json_files, desc="Processing JSON files"):
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        for s in sources:
            if s in file_name:
                size=get_file_size_MB(file_path)
                tokens=tokens_count(file_path,tokenizer)
                for la in languages:
                    if la in file_name:
                        file_size[s][la[1:]]=size
                        token_counts[s][la[1:]]=tokens
                        break

    for item in file_size:
        source_size=0
        for subitem in file_size[item].values():
            source_size+=subitem
        final_file_size[item]=source_size

    for item in token_counts:
        source_tokens=0
        for subitem in token_counts[item].values():
            source_tokens+=subitem
        final_total_tokens[item]=source_tokens

    with open(output_tokens_path,'w',encoding='utf-8') as f:
        json.dump(token_counts,f,ensure_ascii=False,indent=2)
        f.write('\n')
        json.dump(final_total_tokens,f,ensure_ascii=False,indent=2)
    with open(output_size_path,'w',encoding='utf-8') as f:
        json.dump(file_size,f,ensure_ascii=False,indent=2)
        f.write('\n')
        json.dump(final_file_size,f,ensure_ascii=False,indent=2)

    print('File_size')
    print(f'{final_file_size} MB' )
    print('File_tokens')
    print(f'{final_total_tokens} M')











