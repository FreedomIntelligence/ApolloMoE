import argparse
import json
import os
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

question_prompt_en_choice_shot = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:The correct answer is {answer}.<|end_of_text|>
"""
question_prompt_en_choice = """User:You are a medical doctor answering real-world medical exam questions. Select one correct answer from A to D.
Question: {question}
Options:
{options}
Assistant:"""

question_prompt_en_pubmed_shot = """User:You are a medical doctor answering real-world medical exam questions. 
Context: {context} 
Question: {question}
Options:
{options}
Assistant:The correct answer is {answer}.<|end_of_text|>
"""

question_prompt_en_pubmed = """User:You are a medical doctor answering real-world medical exam questions.
Context: {context} 
Question: {question}
Options:
{options}
Assistant:"""

question_prompt_zh_choice_shot = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到E中选择一个正确答案。
问题: {question}
选项:
{options}
Assistant:正确答案是{answer}.<|end_of_text|>
"""

question_prompt_zh_choice = """User:您是一名医生，正在回答现实世界的医学考试问题。请从A到E中选择一个正确答案。
问题: {question}
选项:
{options}
Assistant:"""

question_prompt_es_choice_shot = """User:Usted es un médico que responde preguntas de exámenes médicos del mundo real. Elija una respuesta correcta de la A a la D.
pregunta: {question}
Opciones:
{options}
Assistant:La respuesta correcta es {answer}.<|end_of_text|>
"""
question_prompt_es_choice = """User:Usted es un médico que responde preguntas de exámenes médicos del mundo real. Elija una respuesta correcta de la A a la D.
pregunta: {question}
Opciones:
{options}
Assistant:"""

question_prompt_fr_choice_shot = """User:Vous êtes un médecin et répondez à des questions d'examen médical du monde réel. Veuillez choisir une bonne réponse de A à E.
question: {question}
Possibilités:
{options}
Assistant:La bonne réponse est {answer}.<|end_of_text|>
"""
question_prompt_fr_choice = """User:Vous êtes un médecin et répondez à des questions d'examen médical du monde réel. Veuillez choisir une bonne réponse de A à E.
question: {question}
Possibilités:
{options}
Assistant:"""

question_prompt_hi_choice_shot = """User:आप एक डॉक्टर हैं जो वास्तविक दुनिया की मेडिकल परीक्षा के सवालों का जवाब दे रहे हैं। कृपया A से D तक सही उत्तर चुनें।
सवाल: {question}
विकल्प:
{options}
Assistant:सही उत्तर है{answer}.<|end_of_text|>
"""
question_prompt_hi_choice = """User:आप एक डॉक्टर हैं जो वास्तविक दुनिया की मेडिकल परीक्षा के सवालों का जवाब दे रहे हैं। कृपया A से D तक सही उत्तर चुनें।
सवाल: {question}
विकल्प:
{options}
Assistant:"""

question_prompt_ar_choice_shot = """User:أنت طبيب يجيب على أسئلة الفحص الطبي في العالم الحقيقي. الرجاء اختيار الإجابة الصحيحة من أ إلى د.
سؤال: {question}
خيارات:
{options}
Assistant:{answer}الإجابة الصحيحة هي.<|end_of_text|>
"""
question_prompt_ar_choice = """User:أنت طبيب يجيب على أسئلة الفحص الطبي في العالم الحقيقي. الرجاء اختيار الإجابة الصحيحة من أ إلى د.
سؤال: {question}
خيارات:
{options}
Assistant:"""

question_prompt_ru_choice_shot = """User:Вы врач, отвечающий на реальные вопросы медицинского осмотра. Пожалуйста, выберите правильный ответ от А и B.
вопрос: {question}
Параметры:
{options}
Assistant:Правильный ответ{answer}.<|end_of_text|>
"""
question_prompt_ru_choice = """User:Вы врач, отвечающий на реальные вопросы медицинского осмотра. Пожалуйста, выберите правильный ответ от А и B.
вопрос: {question}
Параметры:
{options}
Assistant:"""

question_prompt_de_choice_shot = """User:Sie sind ein Arzt, der Fragen zu medizinischen Untersuchungen aus der Praxis beantwortet. Bitte wählen Sie eine richtige Antwort von A bis D.
Frage: {question}
Optionen:
{options}
Assistant:Die richtige Antwort ist{answer}.<|end_of_text|>
"""
question_prompt_de_choice = """User:Sie sind ein Arzt, der Fragen zu medizinischen Untersuchungen aus der Praxis beantwortet. Bitte wählen Sie eine richtige Antwort von A bis D.
Frage: {question}
Optionen:
{options}
Assistant:"""

question_prompt_pt_choice_shot = """User:Você é um médico que responde a perguntas de exames médicos do mundo real. Escolha uma resposta correta de A a D.
pergunta: {question}
Opções:
{options}
Assistant:A resposta correta é{answer}.<|end_of_text|>
"""
question_prompt_pt_choice = """User:Você é um médico que responde a perguntas de exames médicos do mundo real. Escolha uma resposta correta de A a D.
pergunta: {question}
Opções:
{options}
Assistant:"""  

question_prompt_it_choice_shot = """User:Sei un medico che risponde a domande sugli esami medici del mondo reale. Scegli una risposta corretta dalla A alla E.
domanda: {question}
Opzioni:
{options}
Assistant:La risposta corretta è{answer}.<|end_of_text|>
"""
question_prompt_it_choice = """User:Sei un medico che risponde a domande sugli esami medici del mondo reale. Scegli una risposta corretta dalla A alla E.
domanda: {question}
Opzioni:
{options}
Assistant:"""


question_prompt_ko_choice_shot = """User:당신은 실제 건강 검진 문제에 답하는 의사입니다. A부터 E까지 정답을 선택해주세요.
질문: {question}
옵션:
{options}
Assistant:정답은{answer}.<|end_of_text|>
"""
question_prompt_ko_choice = """User:당신은 실제 건강 검진 문제에 답하는 의사입니다. A부터 E까지 정답을 선택해주세요.
질문: {question}
옵션:
{options}
Assistant:"""

question_prompt_ja_choice_shot = """User:あなたは現実世界の健康診断の質問に答える医師です。 正しい答えを A ～ E から選択してください。
質問: {question}
オプション:
{options}
Assistant:正しい答えは、{answer}.<|end_of_text|>
"""
question_prompt_ja_choice = """User:あなたは現実世界の健康診断の質問に答える医師です。 正しい答えを A ～ E から選択してください。
質問: {question}
オプション:
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
        if source in ['cmb-single', 'cmexam', 'cmmlu-medical', 'medqa-mcmle']:
            few_shot_prompt = question_prompt_zh_choice_shot
            question_prompt = question_prompt_zh_choice
        elif source in ['medmcqa', 'medqa-usmle', 'mmlu-medical']:
            few_shot_prompt = question_prompt_en_choice_shot
            question_prompt = question_prompt_en_choice
        elif source in ['headqa']:
            few_shot_prompt = question_prompt_es_choice_shot
            question_prompt = question_prompt_es_choice
        elif source in ['frenchmedmcqa']:
            few_shot_prompt = question_prompt_fr_choice_shot
            question_prompt = question_prompt_fr_choice
        elif source in ['mmlu-medical-ar']:
            few_shot_prompt = question_prompt_ar_choice_shot
            question_prompt = question_prompt_ar_choice
        elif source in ['mmlu-medical-hi']:
            few_shot_prompt = question_prompt_hi_choice_shot
            question_prompt = question_prompt_hi_choice
        elif source in ['KorMedMCQA']:
            few_shot_prompt = question_prompt_ko_choice_shot
            question_prompt = question_prompt_ko_choice
        elif source in ['MMLU_college_medicine_pt', 'MMLU_anatomy_pt', 'MMLU_clinical_knowledge_pt',
                         'MMLU_college_biology_pt', 'MMLU_medical_genetics_pt', 'MMLU_professional_medicine_pt']:
            few_shot_prompt = question_prompt_pt_choice_shot
            question_prompt = question_prompt_pt_choice
        elif source in ['MMLU_professional_medicine_de', 'MMLU_college_medicine_de', 'MMLU_college_biology_de', 
                        'MMLU_medical_genetics_de', 'MMLU_clinical_knowledge_de', 'MMLU_anatomy_de']:
            few_shot_prompt = question_prompt_de_choice_shot
            question_prompt = question_prompt_de_choice
        elif source in ['IgakuQA']:
            few_shot_prompt = question_prompt_ja_choice_shot
            question_prompt = question_prompt_ja_choice
        elif source in ['MedExpQA']:
            few_shot_prompt = question_prompt_it_choice_shot
            question_prompt = question_prompt_it_choice
        elif source in ['RuMedDaNet']:
            few_shot_prompt = question_prompt_ru_choice_shot
            question_prompt = question_prompt_ru_choice
        else:
            few_shot_prompt = question_prompt_en_pubmed_shot
            question_prompt = question_prompt_en_pubmed
            
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
                if len(question) + real_question_len + len(sample) < 4096:
                    question += sample
            question += real_question
            if len(question)>4096:
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