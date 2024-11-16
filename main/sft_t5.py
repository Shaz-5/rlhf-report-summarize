#!/usr/bin/env python
# coding: utf-8

# # Supervised Fine-tuning

# In[2]:


# !pip install evaluate transformers[torch] nltk rouge_score sentencepiece tensorboard matplotlib


# In[4]:


import os
import re
import random

import evaluate
import numpy as np
import torch

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


# In[5]:


# seed

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# In[8]:


output_dir = "../models/sft_t5"

# hyperparameters
train_batch_size = 16
gradient_accumulation_steps = 1
learning_rate = 1e-4
eval_batch_size = 1
eval_steps = 100
max_input_length = 512
# save_steps = 300
num_train_epochs = 5
random.seed(42)


# In[9]:


# model - t5

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
print('Loaded T5.')


# In[23]:


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        df = pd.read_csv(data_path)
        
        self.texts = df['findings'].tolist()
        self.summaries = df['impression'].tolist()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):

        text = self.texts[idx]
        summary = self.summaries[idx]
        
        input_text = f"summarize: {text}"
        
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            summary,
            max_length=self.max_length // 2,  # summaries should be shorter
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encodings["input_ids"].squeeze(0)
        attention_mask = input_encodings["attention_mask"].squeeze(0)
        labels = target_encodings["input_ids"].squeeze(0)
        
        # replace padding token id with -100 for labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# In[24]:


# load dataset

train_dataset = SFTDataset(
    '../data/preprocessed/train.csv',
    tokenizer,
    max_length=max_input_length,
)

val_dataset = SFTDataset(
    '../data/preprocessed/val.csv',
    tokenizer,
    max_length=max_input_length,
)


# In[12]:


# rouge metric
rouge = evaluate.load("rouge")


# In[13]:


# def compute_metrics(eval_preds):
#     predictions = eval_preds.predictions
#     labels = eval_preds.label_ids

#     # Decode generated summaries
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
#     print(decoded_preds)
        
#     # Replace -100 in the labels as we can't decode them
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
#     return result


# ## Finetuning

# In[25]:


# trainer

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_accumulation_steps=1,
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_checkpointing=True,
    half_precision_backend=True,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=0,
    load_best_model_at_end=True,
    logging_steps=10,
    logging_dir='../logs/sft_t5',
    log_level='info',
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
    data_collator=default_data_collator,
#     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)


# In[26]:


try:
    trainer.train()
except KeyboardInterrupt:
    print('Training Stopped.')
    
# trainer.save_model(output_dir)


# ## Evaluation

# In[29]:


test_dataset = SFTDataset(
    '../data/preprocessed/test.csv',
    tokenizer,
    max_length=max_input_length,
)

model_path = next((f"../models/sft_t5/{folder}" for folder in os.listdir('../models/sft_t5') if re.search(r'checkpoint-\d+', folder)), None)
model.from_pretrained(model_path)
print('Loaded saved model.')


# In[31]:


trainer_eval_results = trainer.evaluate(test_dataset)

print('Trainer evaluation results on test set: ')
print(trainer_eval_results)


# In[20]:


def run_evaluation(model, test_dataset, tokenizer, batch_size=8, device='cuda'):
    model.eval()
    all_predictions = []
    all_references = []
    
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=150,
            )
            
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # replace -100 with pad_token_id for decoding
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(decoded_preds)
            all_references.extend(decoded_refs)
    
    # ROUGE scores
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_predictions, references=all_references)
    
    return results, all_predictions, all_references


# In[21]:


# evaluation
# model.to('cuda')
eval_results, preds, refs = run_evaluation(model, test_dataset, tokenizer)

print("\nEvaluation Results:")
for metric, score in eval_results.items():
    print(f"{metric}: {score:.4f}")


# In[23]:


import pickle

os.makedirs('../results/sft_t5', exist_ok=True)

# save scores
with open(f'../results/sft_t5/rouge_scores.pkl', 'wb') as f:
    pickle.dump(eval_results, f)

# save predictions
with open(f'../results/sft_t5/predictions.pkl', 'wb') as f:
    pickle.dump(preds, f)

# save references
with open(f'../results/sft_t5/references.pkl', 'wb') as f:
    pickle.dump(refs, f)
    
print('Saved eval results.')

