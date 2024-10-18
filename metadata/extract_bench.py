import os
import json
import argparse

def extract(bench_json_path,cls_json_path,cls):
    with open(bench_json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        extract_data=[item for item in data if item['class']==cls]
    with open(cls_json_path,'w',encoding='utf-8') as f:
        json.dump(extract_data,f,ensure_ascii=False,indent=2)


if __name__=='__main___':
    parser=argparse.ArgumentParser(description='Args of extracting benckmark')
    parser.add_argument('--bench_path',type=str,default='/path/to/ApolloMoEBench.json')
    parser.add_argument('--major_bench_path',type=str,default='/path/to/major_test.json')
    parser.add_argument('--minor_bench_path',type=str,default='/path/to/minor_test.json')
    args=parser.parse_args()
    extract(args.bench_path,args.major_bench_path,'major')
    extract(args.bench_path,args.minor_bench_path,'minor')

            