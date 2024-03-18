from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Honorificity Classification
def formal_classifier(text):
    model = AutoModelForSequenceClassification.from_pretrained("j5ng/kcbert-formal-classifier")
    tokenizer = AutoTokenizer.from_pretrained('j5ng/kcbert-formal-classifier')
    formal_classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    try:
        if formal_classifier(text)[0]['label'] == 'LABEL_0':
            tone = '반말(informal)'
        else    :
            tone = '존댓말(formal)'
    except RuntimeError as e:
            print(e)
            tone = ['반말(informal)', '반말(informal)']
            return 
    return tone