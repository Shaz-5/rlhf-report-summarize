#!/usr/bin/env python
# coding: utf-8

# !pip install transformers[torch] sentencepiece pandas

# # Create Preference Dataset

# In[16]:


import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast  # For FP16 mixed precision

import os
import pandas as pd
from tqdm import tqdm


# In[3]:


# T5 model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Loaded model.')


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[11]:


# dataset class for batch processing
class ReportDataset(Dataset):
    def __init__(self, reports, tokenizer, max_input_length=512):
        self.reports = reports
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        text = self.reports[idx]
        encoding = self.tokenizer(
            "summarize: " + text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        return encoding


# ## Train

# In[4]:


# dataset
train_df = pd.read_csv("../data/preprocessed/train.csv")


# In[12]:


# dataset and dataloader for batch processing

dataset = ReportDataset(train_df['findings'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


# In[13]:


# generate summaries

def generate_batch_summaries(dataloader, model, tokenizer):
    summaries = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            with autocast():  # Enable FP16 mixed precision
                # Generate summaries
                summary_ids = model.generate(input_ids, attention_mask=attention_mask, 
                                             max_length=200, num_beams=4, early_stopping=True)

            # Decode and collect the summaries
            summaries.extend([tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids])
    
    return summaries


# In[ ]:


train_summaries = generate_batch_train_summaries(dataloader, model.to(device), tokenizer)


# In[24]:


train_df['t5_summary'] = train_summaries


# In[25]:


train_df.head()


# In[17]:


os.makedirs('../data/preference_dataset', exist_ok=True)


# In[29]:


train_df.to_csv('../data/preference_dataset/preference_train.csv', index=False)


# ## Val

# In[4]:


# dataset
val_df = pd.read_csv("../data/preprocessed/val.csv")


# In[12]:


# dataset and dataloader for batch processing

dataset = ReportDataset(val_df['findings'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


# In[ ]:


val_summaries = generate_batch_val_summaries(dataloader, model.to(device), tokenizer)


# In[24]:


val_df['t5_summary'] = val_summaries


# In[25]:


val_df.head()


# In[29]:


val_df.to_csv('../data/preference_dataset/preference_val.csv', index=False)


# ## Test

# In[4]:


# dataset
test_df = pd.read_csv("../data/preprocessed/test.csv")


# In[12]:


# dataset and dataloader for batch processing

dataset = ReportDataset(test_df['findings'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


# In[ ]:


test_summaries = generate_batch_test_summaries(dataloader, model.to(device), tokenizer)


# In[24]:


test_df['t5_summary'] = test_summaries


# In[25]:


test_df.head()


# In[29]:


test_df.to_csv('../data/preference_dataset/preference_test.csv', index=False)

