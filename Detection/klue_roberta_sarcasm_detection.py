import io
import json
import datasets
import numpy as np
import pandas as pd
import multiprocessing

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import Trainer
from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

with open("./XXXX-8.json","r") as f:
  json_data = json.load(f)
json.dumps(json_data)

train_df = pd.DataFrame(json_data["train"])
validataion_df = pd.DataFrame(json_data["validation"])
test_df = pd.DataFrame(json_data["test"])

# Model Setting
model_checkpoint = "klue/roberta-large"
batch_size = 8

metric = load_metric("glue", "qnli")
metric_name = "accuracy"

num_labels = 2 # Sarcasm, Non-Sarcasm
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

args = TrainingArguments(
    "test-nli",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Sarcasm Detection Experiment in N-turn Context
def n_turn_detection() :
    
    turn = "random_shuffled_context"
    # Full turn Context is "random_shuffled_context"
    # 3 turn Context is ""random_shuffled_last_three"
    # 2 turn Context is ""random_shuffled_last_two"
    # 1 turn Context is  ""random_shuffled_last_one"
    
    full_train_dataset = Dataset.from_pandas(train_df[['label_for_classification',turn,'random_shuffled_response']])
    full_validataion_dataset = Dataset.from_pandas(validataion_df[['label_for_classification',turn,'random_shuffled_response']])
    full_test_dataset = Dataset.from_pandas(test_df[['label_for_classification',turn,'random_shuffled_response']])

    full_datasets = datasets.DatasetDict({"train":full_train_dataset,
                                    "validation" : full_validataion_dataset,
                                    "test" : full_test_dataset})

    context_key, response_key = (turn, "random_shuffled_response")

    def preprocess_turn_function(examples):
        model_inputs = tokenizer(
            examples[context_key],
            examples[response_key],
            padding = "longest",
            return_token_type_ids=False,
        )
        model_inputs['label'] = [l for l in examples['label_for_classification']]
        return model_inputs
    
    full_encoded_datasets = full_datasets.map(preprocess_turn_function, batched=True)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=full_encoded_datasets["train"],
        eval_dataset=full_encoded_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    output = trainer.predict(full_encoded_datasets['test'])
    
    preds=np.argmax(output.predictions, axis=-1)
    
    full_test_label = full_test_dataset['label_for_classification']
    
    print("Balanced_Accuracy_Score : ",balanced_accuracy_score(full_test_label,preds))
    print(classification_report(full_test_label,preds, digits=4))

# Sarcasm Detection Experiment in Response 
def response_detection() :
    response_train_dataset = Dataset.from_pandas(train_df[['label_for_classification','random_shuffled_response']])
    response_validataion_dataset = Dataset.from_pandas(validataion_df[['label_for_classification','random_shuffled_response']])
    response_test_dataset = Dataset.from_pandas(test_df[['label_for_classification','random_shuffled_response']])

    response_datasets = datasets.DatasetDict({"train":response_train_dataset,
                                    "validation" : response_validataion_dataset,
                                    "test" : response_test_dataset})

    response_key = ("random_shuffled_response")
    
    def preprocess_response_function(examples):
        model_inputs = tokenizer(
            examples[response_key],
            max_length=  128,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
        )
        model_inputs['label'] = [l for l in examples['label_for_classification']]
        return model_inputs
    
    response_encoded_datasets = response_datasets.map(preprocess_response_function, batched=True)

    response_trainer = Trainer(
        model,
        args,
        train_dataset=response_encoded_datasets["train"],
        eval_dataset=response_encoded_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    ) 
    
    response_trainer.train()
    
    response_output = response_trainer.predict(response_encoded_datasets['test'])
    
    response_preds=np.argmax(response_output.predictions, axis=-1)
    response_test_label = response_test_dataset['label_for_classification']

    print("Balanced_Accuracy_Score : ",balanced_accuracy_score(response_test_label,response_preds))
    print(classification_report(response_test_label,response_preds, digits=4))