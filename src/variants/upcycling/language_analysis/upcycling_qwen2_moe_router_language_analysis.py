
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
import argparse
from accelerate import Accelerator
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
from collections import defaultdict
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,tokenizer):
        self.data = []
        with open(data_path) as f:
            self.data = json.load(f)
        dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
        if dist_flag_0:
            print(f'load {len(self.data)} data from {data_path}')
        self.tokenizer = tokenizer
        self.debug = True

    def __getitem__(self, index):
        item = self.data[index]
        return {
            'data': item,
            'input': item['question']
        }

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        batch_query = [x['input'] for x in batch]
        batch_data = [x['data'] for x in batch]
        out_batch = {}
        out_batch['data'] = batch_data
        out_batch['input_ids'] = self.tokenizer(batch_query, return_tensors='pt', padding=True)['input_ids']
        out_batch['attention_mask'] = self.tokenizer(batch_query, return_tensors='pt', padding=True)['attention_mask']
        dist_flag_0 = True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0) else False
        if self.debug and dist_flag_0:
            decoded_texts = self.tokenizer.batch_decode(out_batch['input_ids'], skip_special_tokens=False)
            for idx, sample in enumerate(decoded_texts):
                print(f'*******************batch_texts[{idx}]**********************************')
                print(sample)
            self.debug = False
        return out_batch
        

def count_ratio(input_path,mlp_only_layers):
    
    with open(input_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    layers_count=[]
    layers_count_original=[]
    for layer in data:
        count=[0]*12
        for item in layer:
            for e in item:
                count[e]+=1
        layers_count_original.append(count)
        count=np.array(count)
        #order adjust
        count=count[[2,11,0,3,4,5,10,7,8,9,1,6]].tolist()
        layers_count.append(count)
    output_dir=os.path.dirname(input_path)

    count_original_output_path=os.path.join(output_dir,'count_original.json')
    with open(count_original_output_path,'w',encoding='utf-8') as f:
        json.dump(layers_count_original,f,ensure_ascii=False,indent=2)

    print(f'***********original count of token saved in {count_original_output_path}*************')

    count_output_path=os.path.join(output_dir,'count.json')
    with open(count_output_path,'w',encoding='utf-8') as f:
        json.dump(layers_count,f,ensure_ascii=False,indent=2)

    print(f'***********count of token saved in {count_output_path}*************')

    layers=torch.tensor(layers_count_original)
    layers_ratio=layers/layers.sum(dim=1,keepdims=True)
    if mlp_only_layers:
        layers_ratio[[mlp_only_layers],:]=0
    ratio_original_output_path=os.path.join(output_dir,'ratio_original.json')
    with open(ratio_original_output_path,'w',encoding='utf-8') as f:
        json.dump(layers_ratio.tolist(),f,ensure_ascii=False,indent=2)

    print(f'***********original ratio of token saved in {ratio_original_output_path}*************')

    layers=torch.tensor(layers_count)
    layers_ratio=layers/layers.sum(dim=1,keepdims=True)
    if mlp_only_layers:
        layers_ratio[[mlp_only_layers],:]=0
    ratio_output_path=os.path.join(output_dir,'ratio.json')
    with open(ratio_output_path,'w',encoding='utf-8') as f:
        json.dump(layers_ratio.tolist(),f,ensure_ascii=False,indent=2)
        
    print(f'***********ratio of token saved in {ratio_output_path}*************')

    return ratio_output_path


def plot(input_path):
    title_name='Mix' if os.path.dirname(input_path)[-2:]=='ix' else os.path.dirname(input_path)[-2:]
    output_path=os.path.join(os.path.dirname(input_path),f'{title_name}.png')

    # print(output_path)
    plt.figure(figsize=(30, 30))
    with open(input_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        np_data=np.array(data)
        sns.set(font_scale=3)
        ax=sns.heatmap(data=np_data,annot=True,cmap='OrRd',vmin=0,vmax=1,fmt='.2f',square=False,cbar=False,
                    # xticklabels=['ar','de','en','es','fr','hi','it','ja','ko','pt','ru','zh'],
                    xticklabels=['en','zh','ar','es','fr','hi','ru','ja','ko','pt','de','it'],
                    yticklabels=[i+1 for i in range(np_data.shape[0])])
        ax.invert_yaxis()
        
        plt.title(title_name.upper(),fontweight='bold',fontsize=50,pad=20)
        plt.xticks(fontsize=45)
        plt.yticks(rotation=0,fontsize=40)
        plt.xlabel('Experts',fontweight='bold',fontsize=45,labelpad=10)  
        plt.ylabel('Decoder Layers',fontweight='bold',fontsize=45,labelpad=10)
        rect = ax.patch
        rect.set_edgecolor('black') 
        rect.set_linewidth(2)
    # plt.show()
    plt.savefig(output_path, dpi=300,bbox_inches='tight')
    print(f'***********token routing pictures saved in {output_path}*************')


def language_analysis(args):
    accelerator = Accelerator()
    
    model_path = args.model_path
    accelerator.print(f'****************model_path:{model_path}******************')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True)#.half()
    # generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id, num_return_sequences=args.num_return, max_new_tokens=256, min_new_tokens=2, do_sample=False, temperature=1.0, top_k=50, top_p=1.0)
    config=model.config
    mlp_only_layers=config.mlp_only_layers

    dataset = TestDataset(args.input_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    model = model.eval()
    if dist.is_initialized():
        accelerator.print(f'****************dist.get_world_size():{dist.get_world_size()}******************')

    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.print(f'******************load_model from {model_path}******************')

    if accelerator.is_main_process:
        fp = open(args.output_path,'w')
        
    dataloader_iterator = tqdm(dataloader, total=len(dataloader)) if accelerator.is_main_process else dataloader

    data=[]
    device=torch.cuda.current_device()
    for batch in dataloader_iterator:
        batch_input_ids = batch["input_ids"]
        batch_attention_mask=batch['attention_mask']

        outputs=model(input_ids=batch_input_ids,attention_mask=batch_attention_mask,return_dict=True,use_cache=False)
        
        batch_token_routing=outputs.token_routing.tolist()

        if dist.is_initialized():
            all_batch_token_routing=[None]*dist.get_world_size()
            dist.all_gather_object(all_batch_token_routing,batch_token_routing)  
            # accelerator.print(len(all_batch_token_routing),type(all_batch_token_routing[0]))
        else:
            all_batch_token_routing=[batch_token_routing,]
        
        for item in all_batch_token_routing:
            data.append(item)
    
    if accelerator.is_main_process:
        final_tensor=torch.cat([torch.tensor(item).to(device) for item in data],dim=1)
        print(final_tensor.shape)
        fp.write(json.dumps(final_tensor.tolist(), ensure_ascii=False))
        fp.flush()

    accelerator.wait_for_everyone()
    return mlp_only_layers
    # print(f'***********routing_result saved in {args.output_path}*************')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("--input_path", type=str, help="path to the input data")
    parser.add_argument("--output_path", type=str, help="path to the output data")
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()
    mlp_only_layers=language_analysis(args)

    dist_flag_0=True if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank()==0) else False
    if dist_flag_0:
        ratio_path=count_ratio(args.output_path,mlp_only_layers)
        plot(ratio_path)


'''

accelerate launch ./src/evaluate/eval_qwen.py \
--model_path=/sds_wangby/models/Qwen-1_8B \
--input_path=./data/Qwen-1.8B/test.json \
--output_path=./result/Qwen1.8B/model_ans.jsonl \
--score_path=./result/Qwen1.8B/score.json \
--num_return=1 \
--batch_size=8 > ${log_folder}/$log_name 2>&1 &

'''
