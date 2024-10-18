import json
import os
from collections import defaultdict
import argparse

def extract(dataset_path,major_path,minor_path):
    with open(dataset_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        major_dict=defaultdict(list)
        minor_dict=defaultdict(list)
        for item in data:
            content=[subitem['value'] for subitem in item['conversations']]
            if item['class']=='major':
                major_dict[item['source']].append(content)
            elif item['class']=='minor':
                minor_dict[item['source']].append(content)

    with open(major_path,'w',encoding='utf-8') as f:
        json.dump(major_dict,f,ensure_ascii=False,indent=2)
    with open(minor_path,'w',encoding='utf-8') as f:
        json.dump(minor_dict,f,ensure_ascii=False,indent=2)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Args of extracting dataset')
    parser.add_argument('--dataset_path',type=str,default='/path/to/ApolloMoEDataset_0_1.json')
    parser.add_argument('--major_dataset_path',type=str,default='/path/to/MajorDataset.json')
    parser.add_argument('--minor_dataset_path',type=str,default='/path/to/MinorDataset.json')
    args=parser.parse_args()
    extract(args.dataset_path,args.major_dataset_path,args.minor_dataset_path)