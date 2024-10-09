import json 
import os
import pandas as pd 
import torch

def results(source_path,root,output):
    with open(source_path,'r',encoding='utf-8') as f:
        source=json.load(f)
    tests=['Qwen2-7B_Test','Apollo-MoE-0.5B_Test']
    output_table=dict()
    excel_name='Major_Test'
    for test in tests:
        dir_name=f'{test}'

        files=os.listdir(os.path.join(root,dir_name))
        score_en,score_zh=0,0
        total_de,right_de=0,0
        total_pt,right_pt=0,0
        total_fr,right_fr=0,0
        total_it,right_it=0,0
        for file in files:
            with open(os.path.join(root,dir_name,'all.json'),'r',encoding='utf-8') as f:
                total=json.load(f)
            with open(os.path.join(root,dir_name,'right.json'),'r',encoding='utf-8') as f:
                right=json.load(f)
            if file=='score.json':
                with open(os.path.join(root,dir_name,file),'r',encoding='utf-8') as f:
                    score=json.load(f)
                temp=dict()
                for k,v in source.items():
                    if isinstance(v,list):
                        for bm in v:
                            temp[bm]=score[bm]
                            if bm in ['medqa-usmle','pubmedqa','mmlu-medical','medmcqa']:
                                score_en+=score[bm]                        
                            if bm in ["cmb-single","medqa-mcmle","cmmlu-medical","cmexam"]:
                                score_zh+=score[bm]
                            if bm in ['MMLU_anatomy_de','MMLU_professional_medicine_de','MMLU_clinical_knowledge_de','MMLU_medical_genetics_de','MMLU_college_biology_de','MMLU_college_medicine_de']:
                                total_de+=total[bm]
                                right_de+=right[bm]
                            if bm in ['MMLU_anatomy_pt','MMLU_professional_medicine_pt','MMLU_clinical_knowledge_pt','MMLU_medical_genetics_pt','MMLU_college_biology_pt','MMLU_medical_genetics_pt']:
                                total_pt+=total[bm]
                                right_pt+=right[bm]
                            if bm in ['frenchmedmcqa','mmlu-medical-fr']:
                                total_fr+=total[bm]
                                right_fr+=right[bm]
                            if bm in ['MedExpQA','mmlu-medical-it']:
                                total_it+=total[bm]
                                right_it+=right[bm]
                    else:raise ValueError
                    if k=='en_source':
                        temp['average-en']=score_en/4
                    if k=='zh_source':
                        temp['average-zh']=score_zh/4
                    if k=='pt_source':
                        temp['weighted-average-pt']=right_pt/total_pt
                    if k=='de_source':
                        temp['weighted-average-de']=right_de/total_de
                    if k=='fr_source':
                        temp['weighted-average-fr']=right_fr/total_fr
                    if k=='it_source':
                        temp['weighted-average-it']=right_it/total_it

        output_table[f'{dir_name}']=temp
    print(output_table)
    df=pd.DataFrame(output_table)
    df=df.T
    df=df.loc[:,['mmlu-medical-ar','weighted-average-de','average-en','headqa','weighted-average-fr','mmlu-medical-hi','weighted-average-it','IgakuQA','KorMedMCQA','weighted-average-pt','RuMedDaNet','average-zh']]
    df.to_excel(f'{excel_name}.xlsx')


source_path='path/to/utils/source_new.json'
root='/path/to/results'

results(source_path,root,'.')