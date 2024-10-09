import os
os.environ['HF_HOME'] = 'D:/0_huggingface/huggingface_test'


from datasets import load_dataset
raw_datasets = load_dataset("FreedomIntelligence/Huatuo26M-GPTShine")
            
print(raw_datasets)

