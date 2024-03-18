import os, sys, json, csv
import pandas as pd
import openai 

from tqdm import tqdm

def moderation_create(inputs):
    response = openai.Moderation.create(
    input=inputs
    )
    return response['results'][0]['flagged']


file_path = "PreProcess_Dataset"
save_path = "Moderation_result"
file_list = os.listdir(file_path)
for file_ in tqdm(file_list):
    data_path = os.path.join(file_path, file_)
    data = pd.read_csv(data_path)
    result_list = []
    for idx, row in tqdm(zip(data['file_name'], data['sarcasm_generation'])):
        result_list.append({"file_name":idx, "result": moderation_create(row)})
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(os.path.join(save_path, file_[:-4]) + "Moderated.csv")