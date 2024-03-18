import json, os, sys, re
import pandas as pd
import openai
import time

from time import sleep
from tqdm import tqdm
from utils import *

from Generation import *

def conversation_abnormal_detection(text):
    
    messages = [  
        {'role': 'system', 'content': f'''You are really good at evaluating the dialogue.
Task Description:  한국어 대화를 잘 이해하는 사람으로써 한국어 맥락에 집중해 제공하는 대화가 어색한지 판별해줘.
대화가 이어지지 않는 경우에는 1, 대화가 자연스러운 경우에는 0으로 숫자만 이용해서 대답해줘.'''},
         
            # example 1
            {'role': 'user', 'content': '''Conversation : 
            A: 엄마 아빠가 다낭에 가있는 동안, 창문을 다 찹쌀떡으로 막아버렸어.
            B: 왜 그랬어? 이유가 뭐야?
            A: 외풍이 너무 심해서 따뜻하게 하려고 했는데, 이제 창문이 열리질 않아.}'''},      
            {'role': 'assistant', 'content': '1'},               
            
            # example 2
            {'role': 'user', 'content': '''Conversation : 
            A: 요즘 다이어트 어떻게 돼가?
            B: 음.. 아직은 잘 모르겠어. 먹는 걸 줄이기가 힘들더라고.
            A: 그럼 쥬비스같은 곳에서 식단관리 받아보는 건 어때?
            B: 그래서 가봤는데, 너무 비싸더라고.}'''},       
            {'role': 'assistant', 'content': '0'},  
            
            # example 3
            {'role': 'user', 'content': '''Conversation : 
            A: 헐, 이번 주말에 그 영화 보려고 했는데 티켓이 다 매진되었대.
            B: 진짜? 왜 갑자기 그렇게 된 거지?
            A: 처음에는 별로 인기 없었는데 어느 순간부터 인기가 떨어졌대.}'''},      
            {'role': 'assistant', 'content': f'''1'''},
            
            # prompt
            {'role': 'user', 'content': "Conversation : \"\"\"\n"+text+"\"\"\"\n}"},
            ]    
        
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature = 1.0,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0,
    )

    sarcasm = (str(response['choices'][0]['message']['content']).strip())
    
    return sarcasm

data = pd.read_csv("KoCoSa_Dataset.csv")

print(f"===========================Abnormal Detection Start====================================")    
result = []    
for idx in tqdm(range(0, len(data))):
    dialog = data['sarcasm_generation_spell_checked'][idx]
    _, sample = dialog_preprocessing(dialog)
    try:
        abnormal_result = conversation_abnormal_detection(sample)
        result.append({
            "File":data['File'][idx],
            "sarcasm_generation_spell_checked": sample,
            "explanation": data['explanation'][idx],
            "Sarcasm(Y/N)": data['Sarcasm(Y/N)'][idx],
            "Conversation": data['Conversation'][idx],
            "GPT-Abnormal": abnormal_result,
        })
    except (openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
        print("API Error occured: ", str(e))
        print("=======================Retry in 300 seconds =========================")
        sleep(300)
        continue
    
    except Exception as e:
        print("An error occurred:", str(e))
        result_df = pd.DataFrame(result)
        result_df.to_csv(f"abnoraml_detection_result{idx}.csv")
        print(time.strftime('%Y.%m.%d - %H:%M:%S'))
        print("=========================STOP Generating: ERROR=========================")
        print(f"========================Final Point: {idx}========================")
        sys.exit()
        
    if idx%100 == 0 and idx != 0:
        result_df = pd.DataFrame(result)
        result_df.to_csv(f"abnormal_result/abnoraml_detection_result{idx//100}.csv")
        print(f"=================={idx//100}_th Is Save!=====================")
        result = []
        sleep(20)
        
result_df = pd.DataFrame(result)
result_df.to_csv(f"abnoraml_detection_result_tempt{idx}.csv")