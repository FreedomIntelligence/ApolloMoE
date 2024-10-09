import argparse
import json
import os
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


question_prompt_en_choice_shot = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:The correct answer is {answer}.<|endoftext|>
"""
question_prompt_en_choice = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:"""



def preprocess(args):
    data_final = []
    with open(args.data_path, 'r') as file:
        data = json.load(file)
    grouped_items = {}
    for item in data:
        source = item.get("source")
        if source not in grouped_items:
            grouped_items[source] = []
        grouped_items[source].append(item)

    for source, items in grouped_items.items():
        debug = True 
        print(f'*********************{source}****************************')
        few_shot_prompt = question_prompt_en_choice_shot
        question_prompt = question_prompt_en_choice
            
        for item in items:
            random_samples = random.sample(items, args.few_shot+1)
            question = ''
            tmp_dict = {}
            # in case item in random_samples
            if item in random_samples:
                random_samples.remove(item)
            else:
                random_samples = random_samples[:-1]
            real_question = question_prompt.format(**item)
            real_question_len = len(real_question)
            for sample in random_samples:
                sample = few_shot_prompt.format(**sample)
                if len(question) + real_question_len + len(sample) < 2048:
                    question += sample
            question += real_question
            if len(question)>2048:
                continue
            if debug:
                print(question)
                debug=False
            
            tmp_dict['source_question'] = item['question']
            tmp_dict['source_option'] = item['options']
            tmp_dict['question'] = question
            tmp_dict['answer'] = item['answer'][1]
            tmp_dict['source'] = item['source']
            data_final.append(tmp_dict)
                
    with open(args.save_path, 'w', encoding='utf-8') as file:
        json.dump(data_final, file, ensure_ascii=False, indent=2)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    # Model Args
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--few_shot', default='', type=int)
    args = parser.parse_args()

    preprocess(args)  