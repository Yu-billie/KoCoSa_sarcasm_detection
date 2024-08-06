import openai

# Sarcastic Dialogue Generation
def generate_sarcasm_with_topic(tone: list, topic, intimacy):  
    A_tone, B_tone = tone 

    messages = [  
        {'role': 'system', 'content': f'You are Korean. You create natural Korean conversations proficiently. Please consider the tone. TONE: A-{A_tone}, B-{B_tone}.'},
        {'role': 'system', 'content': f'''Task Description: 
Sarcasm : someone says something but means the opposite in a mocking or ironic way, often using tone and context to convey the real meaning.
Task Description: Create a completely new Korean conversation related to the provided summary. Then generate a sarcastic sentence in response to the final utterance of the conversation.            
Provide an explanation of how to response sarcastically to the generated conversation. Then write the sarcastic response(about 10 to 15 words) without any additional context.\n


Example 1.(TOPIC: 저녁 메뉴-계란 프라이를 태워 먹지 못하는 상황, TONE: A-반말(Informal), B-반말(Informal))
Intimacy: 4
A: 요리는 잘 돼가?
B: 응 지금까지는 순항중이야. 하나만 빼고.
A: 뭐가 문제야? 잘 안되는게 있어?
B: 계란 후라이가 조금 탔어.
Sarcasm explanation: 계란프라이가 바싹 타버렸다는 마지막 A의 말에 실제로는 부정적인 상황인데, 이 상황을 긍정적인 방향으로 비꼬아 말한다.
Sarcastic response(A): 이거 정말 바삭바삭하겠는걸.

                                    
Example 2.(TOPIC: 자기계발-퇴근 후 자기계발을 위해 학원에 등록한 상황, TONE: A-존댓말(Formal), B-반말(Informal))
Intimacy: 3
A: 퇴근하고 뭐 하는거 있어요?
B: 아니 퇴근하면 힘들잖아. 그냥 집에 가서 쉬어야지.
A: 저는 얼마 전에 영어학원 등록했어요.
B: 아 진짜? 영어 공부 하려고?? 저번 달에는 중국어 공부할거라며?
A: 중국어는 너무 어렵더라고요. 그래서 큰 돈주고 영어학원 다시 등록했어요.
Sarcasm explanation: 영어학원에 등록만 하고 가지 않을 것 같은 상대방의 행동을 긍정적인 기부를 하는 것처럼 비꼬아 말한다.
Sarcastic response(B): 학원에 그렇게 기부를 많이 해도 되는거야?


'''},
        {'role': 'user', 'content': f"TOPIC: {topic}, TONE: TONE: A-{A_tone}, B-{B_tone} \n Intimacy: {intimacy}\nGenerate Example:  "}
    ]    

    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages = messages,
        max_tokens=3000,
        n=1,
        stop=None,
        temperature= 1.1,
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=0
    )
    
    if type(response) == tuple: # when sarcasm response comes out as a tuple
        return response[1] 
    else:
        return response
