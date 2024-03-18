import os
import re
import torch
import openai
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from requests.exceptions import ConnectionError
from soynlp.normalizer import repeat_normalize
from statistics import mean
from time import time, sleep

# Directory
os.chdir('/home/XXXX-1/KoCoSa/')

# Set GPU env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
print(torch.cuda.is_available(), device)

data = pd.read_excel('data/XXXX-7.xlsx')
annotation = data['label'].tolist()
labels = [1 if label == 1 else 0 for label in annotation]  # `sarcasm`==1, `non_sarcasm`, `abnormal`==0
len(labels)

def dialog_preprocessing(input_text):
    sentences = input_text.split('\n')
    conversation = [line for line in sentences if line and 'Sarcasm explanation' not in line and 'Sarcastic response' not in line]
    converted_form = '\n'.join(conversation)
    try:
        match = re.search(r'\(A\): (.*)', ''.join(sentences[-1]))
        response = f'A: {match.group(1)}'
    except:
        match = re.search(r'\(B\): (.*)', ''.join(sentences[-1]))
        response = f'B: {match.group(1)}'

    sample = f"{converted_form}\n{response}"   # context + sarcastic response
    context = converted_form                   # context only

    return sample, context

# Detection Label: sarcasm = 1 / non_sarcasm = 0
def sarcasm_detection_zero(generated_sample):
    system_prompt = """Task Description: You are really good at detecting the sarcastic response at the last utterance of the given dialog.
If the last utterance is sarcastic, print "1". If not sarcastic, print "0". """

    user_prompt = f"""given dialog: {generated_sample}
    Detection Result:
            """

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo', messages=messages,
        temperature = 0.0 ,top_p = 0.8, max_tokens = 300, frequency_penalty=0, presence_penalty=0)

    detection_label = str(response['choices'][0]['message']['content'])
    global completion_tokens_d,prompt_tokens_d
    completion_tokens_d,prompt_tokens_d=response['usage']['completion_tokens'],response['usage']['prompt_tokens']

    return detection_label

# Detection Label: sarcasm = 1 / non_sarcasm = 0
def sarcasm_detection_4shot(generated_sample):
    system_prompt = """Task Description: You are really good at detecting the sarcastic response at the last utterance of the given dialog.
                        If the last utterance is sarcastic, print "1". If not sarcastic, print "0"

                        Example 1:
                        "A: 요리는 잘 되가?
                        B: 응 지금까지는 순항 중이야. 하나만 빼고.
                        A: 뭐가 문제야? 잘 안 되는 게 있어?
                        B: 계란 후라이가 조금 탔어.
                        A: 이거 정말 바삭바삭하겠는걸."
                        Detection Result: 1

                        Example 2:
                        "A: 퇴근하고 뭐 하는 거 있어요?
                        B: 아니 퇴근하면 힘들잖아. 그냥 집에 가서 쉬어야지.
                        A: 저는 얼마 전에 영어학원 등록했어요.
                        B: 아 진짜? 영어공부 하려고?? 저번 달에는 중국어 공부할거라며?
                        A: 중국어는 너무 어렵더라고요. 그래서 큰 돈 주고 영어학원 다시 등록했어요."
                        Detection Result: 0

                        Example 3:
                        "A: 어제 하루 종일 잠만 자느라 시험공부 하나도 못 했어.
                        B: 정말 성실한 하루를 보냈구나. 잘하는 짓이다. "
                        Detection Result: 1

                        Example 4:
                        "A: 왜 그렇게 화난 표정이야?
                        B: 아, 또 그러지 말라니까. 이해가 안 돼?
                        A: 뭐가 그렇게 힘들고 속상한 건데?
                        B: 일이 너무 힘들고, 집안 사정도 복잡해. 무엇보다는 내 마음이 참 괴로워.
                        A: 이제 잠깐 쉬어보면 어때? 좋은 일이 분명 있을거야.
                        B: 어차피 내가 아무리 힘들어도 상황이 바뀌는 것은 없을 거야."
                        Detection Result: 0
                        """

    user_prompt = f"""given dialog: {generated_sample}
    Detection Result:
            """

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo', messages=messages,
        temperature = 0.0 ,top_p = 0.8, max_tokens = 1000, frequency_penalty=0, presence_penalty=0)

    detection_label = str(response['choices'][0]['message']['content'])
    global completion_tokens_d,prompt_tokens_d
    completion_tokens_d,prompt_tokens_d=response['usage']['completion_tokens'],response['usage']['prompt_tokens']

    return detection_label

def sarcasm_detection_8shot(generated_sample):  
    system_prompt = """Task Description: You are really good at detecting the sarcastic response at the last utterance of the given dialog. 
                    If the last utterance is sarcastic, print "1". If not sarcastic, print "0" 
                    
                    Example 1:
                    "A: 요리는 잘 되가?
                    B: 응 지금까지는 순항 중이야. 하나만 빼고.
                    A: 뭐가 문제야? 잘 안 되는 게 있어?
                    B: 계란 후라이가 조금 탔어.
                    A: 이거 정말 바삭바삭하겠는걸." 
                    Detection Result: 1
                                                        
                    Example 2: 
                    "A: 퇴근하고 뭐 하는 거 있어요?
                    B: 아니 퇴근하면 힘들잖아. 그냥 집에 가서 쉬어야지.
                    A: 저는 얼마 전에 영어학원 등록했어요.
                    B: 아 진짜? 영어공부 하려고?? 저번 달에는 중국어 공부할거라며?
                    A: 중국어는 너무 어렵더라고요. 그래서 큰 돈 주고 영어학원 다시 등록했어요."
                    Detection Result: 0
                    
                    Example 3:
                    "A: 어제 하루 종일 잠만 자느라 시험공부 하나도 못 했어. 
                    B: 정말 성실한 하루를 보냈구나. 잘하는 짓이다. "
                    Detection Result: 1 
                    
                    Example 4: 
                    "A: 왜 그렇게 화난 표정이야?
                    B: 아, 또 그러지 말라니까. 이해가 안 돼?
                    A: 뭐가 그렇게 힘들고 속상한 건데?
                    B: 일이 너무 힘들고, 집안 사정도 복잡해. 무엇보다는 내 마음이 참 괴로워.
                    A: 이제 잠깐 쉬어보면 어때? 좋은 일이 분명 있을거야.
                    B: 어차피 내가 아무리 힘들어도 상황이 바뀌는 것은 없을 거야."
                    Detection Result: 0 
                    
                    Example 5:
                    "A: name1아, 오늘 학교에서 시험은 잘 봤니? 
                    B: 사실 어제 하루 종일 자느라 시험 공부를 하나도 못 한 채로 봤어요. 
                    A: 정말 성실한 하루를 보냈구나. 1등도 문제없을 정도야."  
                    Detection Result: 1 
                    
                    Example 6:
                    "A: 오늘 무슨 날이야? 차려입고 왔네? 표정은 또 왜 이렇게 초조해 보여?  
                    B: 오늘 중요한 미팅 날인데, 팀장님이 휴가 내시고 1주일째 연락이 없어. 참 큰 일이야." 
                    Detection Result: 0 
                    
                    Example 7:
                    "A: 또 여행 유튜브 봐? 
                    B: 응 이번 방학도 여행 가긴 글렀어. 지금 해탈한 상태야. 영상이라도 봐야지.
                    A: 나 다음주에 하와이 가는데. 안 됐다. 
                    B: 오~ 정말 하나도 안 부러운 걸~"
                    Detection Result: 1 
                    
                    Example 8:
                    "A: 생일 축하해! 사실 너를 위해 새벽부터 일어나서 수제 케이크를 만들었어. 
                    B: 감동이야. 이걸 혼자 다 만들다니 참 대단하다."
                    Detection Result: 0  
                    """
    
    user_prompt = f"""given dialog: {generated_sample} 
    Detection Result: 
            """
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(model = 'gpt-4', messages=messages,
        temperature = 0.0 ,top_p = 0.8, max_tokens = 1000, frequency_penalty=0, presence_penalty=0) 

    detection_label = str(response['choices'][0]['message']['content'])
    global completion_tokens_d,prompt_tokens_d
    completion_tokens_d,prompt_tokens_d=response['usage']['completion_tokens'],response['usage']['prompt_tokens'] 

    return detection_label  

detected_text = []
detected_label = []
predictions = []
completion_token_sarcasm_detection = []
prompt_token_sarcasm_detection = []
output_list = []
current_idx = 0

for i in range(len(labels)):
    try:
        input_text = data['sarcasm_generation_spell_checked'][i]
        sample, context = dialog_preprocessing(input_text)
        result = sarcasm_detection_4shot(sample)             # sarcasm detection 4-shot
        category = int(result)
        print(f'순서:{i+1}\nTrue Label:{labels[i]}, Annotation:{annotation[i]}\n{result}\n{sample}\n')

        detected_text.append(sample)
        detected_label.append(labels[i])
        predictions.append(category)
        completion_token_sarcasm_detection.append(completion_tokens_d)
        prompt_token_sarcasm_detection.append(prompt_tokens_d)

        current_idx = i+1

    except (openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
        print("API Error occured: ", str(e))
        sleep(600)
        i = current_idx - 1

    output_list.append([detected_text,detected_label,predictions, prompt_token_sarcasm_detection, completion_token_sarcasm_detection])

print(i, len(detected_label))

y_true, y_pred = detected_label, predictions    # Detect ALL
print(balanced_accuracy_score(y_true, y_pred))
report = classification_report(y_true, y_pred)
print(report)

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')

outpath = './result/'
df = pd.DataFrame(output_list)
df.columns = ['detected_text','true_label','predictions','prompt_token_sarcasm_detection','completion_token_sarcasm_detection']

writer = pd.ExcelWriter(outpath + 'gpt35_4shot_sarcasmdetection.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='gpt35_4shot_sarcasmdetection', index=False)
writer.close()

len(output_list)
