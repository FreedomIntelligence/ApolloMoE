import json

def merge(input1,input2,output):
    with open(input1,'r',encoding='utf-8') as f:
        data1=json.load(f)
    with open(input2,'r',encoding='utf-8') as f:
        data2=json.load(f)
    data1.extend(data2)
    with open(output,'w',encoding='utf-8') as w:
        json.dump(data1,w,ensure_ascii=False,indent=2)

def size(input1):
    with open(input1,'r',encoding='utf-8') as f:
        data1=json.load(f)
        print(len(data1))
