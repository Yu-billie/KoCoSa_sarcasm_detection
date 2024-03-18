import random
import openai
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Situation Extraction
def situation_extraction(conversation): 
    
    #* Select start point of corpus for Situation Extraction
    start_point = random.randint(0, len(conversation)-6)
    
    #* Generate Conversation (A, B Speaker tagging)
    conv = ""
    for idx_, sent in enumerate(conversation[start_point:start_point+6]):
        if idx_ % 2 == 0:
            conv += "A: " + sent + "\n"
        else:
            conv += "B: " + sent + "\n"
    messages = [  
        {'role': 'system', 'content': f''' 
Task Description: You are really good at extracting the topic of a conversation. Please extract the topic from the given conversation in Korean.
Please refer to the example below to extract the topics. The topic consists of one major theme and some minor themes.
                                    
Given Conversation:
A: 와 오늘 날씨 진짜 좋다.
B: 그러게, 덥지도 않고 괜찮은 것 같아.
A: 내일은 뭐해?
B: 아마 그냥 집에 있을 것 같아.
A: 이렇게 좋은 날에 집에만 있기 아까울 것 같은데!
B: 그러게. 어딜 나가야 할까?
                                    
TOPIC: 날씨-날씨가 좋아 외출 계획을 세움
                                    
Given Conversation:
Original Conversation 2:
A: 계란 프라이 태웠어.
B: 그럼 우리 저녁 못 먹어?

TOPIC: 저녁 메뉴-계란 프라이를 태워 먹지 못하는 상황
                                    '''},
        {'role': 'system', 'content': f"Given Conversation: \n {conv} \n\n Please summarize the above conversation"},
    ]    

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages = messages,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature= 1.0, 
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if type(response) == tuple: 
        return conv, response[1]
    else:
        return conv, response
    