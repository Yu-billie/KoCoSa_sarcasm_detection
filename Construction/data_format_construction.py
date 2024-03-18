
import Situation_Extraction
import Honorificity_Classification
import Sarcastic_Dialogue_Generation

import os, sys, json
import pandas as pd
import tqdm
import re

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Make Dialogue Format
def make_conv(utterance: list) -> list:
    text = []
    if type(utterance) is not list:
        raise ValueError
    else:
        for idx, utter in enumerate(utterance):
        
            if "speaker_id" not in utter.keys():
                raise KeyError
            else:
                speaker = utter['speaker_id'] 
                conv = utter["form"]
                if idx == 0:
                    text.append(conv)
                else:
                    temp = utterance[idx-1]['speaker_id'] 
                    if temp == speaker: 
                        text[-1] += " " + conv 
                    else:
                        text.append(conv)
        return text 

# Make Dialogue Datset
def make_dataset() :
    
    # NIKL_MESSENGER_v2.0 is National Institute of Korean Language Corpus
    file_list = os.listdir("NIKL_MESSENGER_v2.0")[:30]

    df = pd.DataFrame(columns = ['File_name', 'Given_conversation', 'tone', 'Summarized_topic', 'Generated_conversation', 'Summarize_in_token', 'Summarize_out_token', 'sarcasm_in_token', 'sarcasm_out_token'])
    for idx, file_ in tqdm(enumerate(file_list[10:30])):
        #* data load
        with open(f"NIKL_MESSENGER_v2.0/{file_}", 'r') as f:
            data = json.load(f)
        conversation = make_conv(data['document'][0]['utterance'])
        given_conversation, sum_response = Situation_Extraction.situation_extraction(data, conversation=conversation)

        #* Check Honorificity
        given_conversation2 = given_conversation.split('\n')
        A_conv = re.sub("A: ", "", " ".join(given_conversation2[0::2])).strip()
        B_conv = re.sub("B: ", "", " ".join(given_conversation2[1::2])).strip()
        tone = [Honorificity_Classification.formal_classifier(A_conv), Honorificity_Classification.formal_classifier(B_conv)] #* tone -> list 

        #* Intimacy
        Intimacy = data["document"][0]["metadata"]["setting"]["intimacy"]

        sarcasm_response = Sarcastic_Dialogue_Generation.generate_sarcasm_with_topic(tone, sum_response['choices'][0]['message']['content'], intimacy = Intimacy)

        df.loc[idx] = [file_,
                       given_conversation, tone,
                       sum_response['choices'][0]['message']['content'],
                       sarcasm_response['choices'][0]['message']['content'],
                       sum_response['usage']['prompt_tokens'],
                       sum_response['usage']['completion_tokens'],
                       sarcasm_response['usage']['prompt_tokens'],
                       sarcasm_response['usage']['completion_tokens']]