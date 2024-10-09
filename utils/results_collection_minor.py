import json 
import os
import pandas as pd 
import torch

def results(root,output):
    model_tests=['Qwen2-7B_minorTest','Apollo-MoE-0.5B_minorTest']
    output_table=dict()
    excel_name = f'Minor_Test'
    for la in model_tests:
        dir_name=f'{la}'
        files=os.listdir(os.path.join(root,dir_name))

        for file in files:
            if file=='score.json':
                with open(os.path.join(root,dir_name,file),'r',encoding='utf-8') as f:
                    score=json.load(f)

                temp=dict()
                for k,v in score.items():
                    if k[-3]=='-':
                        temp[k[-2:]]=v
                    if k[-4]=='-':
                        temp[k[-3:]]=v

        output_table[f'{dir_name}']=temp
    print(output_table)
    df=pd.DataFrame(output_table)
    df=df.T
    print(df)
    
    df.to_excel(f'{excel_name}.xlsx')

source_path='path/to/utils/source_new.json'
root='/path/to/results'

results(root,'.')




