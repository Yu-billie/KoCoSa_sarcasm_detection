import os
import sys
import re
import openai
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def dialog_preprocessing(input_text, sarcasm_data = True):
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

    return context, sample#context, response, #sample, context+'\n[SAR]'+response, '\n[SAR]'+response 

def check_length(folder_path):
    file_list = os.listdir(folder_path)
    cnt = 0
    for file_ in file_list:
        data = pd.read_csv(os.path.join(folder_path, file_), index_col=None)
        cnt += len(data)
    return cnt


def check_True(folder_path):
    file_list = os.listdir(folder_path)
    cnt = 0
    result = []
    for file_ in file_list:
        data = pd.read_csv(os.path.join(folder_path, file_), index_col=None)
        if 'result' in data.columns:
            for idx, check in enumerate(data['result']):
                if check:
                    cnt+=1
                    result.append(data['file_name'][idx])
        else:
            sys.exit()
    return result, cnt

def main():
    return None

if __name__ == "__main__":
    main()