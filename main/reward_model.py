#!/usr/bin/env python
# coding: utf-8

# # Reward Model Training

# In[1]:


# !pip install tensorboard transformers[torch] sentencepiece pandas


# In[2]:


import os
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

import pandas as pd
from tqdm import tqdm


# In[3]:


import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5RewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.config = model.config
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.PAD_ID = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        return_dict=False,
    ):
        bs = input_ids.shape[0] // 2
        
        # Get encoder outputs for all sequences
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get decoder outputs for all sequences
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        # Get rewards from decoder outputs
        rewards = self.v_head(decoder_outputs.last_hidden_state).squeeze(-1)
        
        # Split rewards into chosen and rejected
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Get end scores (use the last non-padding token)
        chosen_end_scores = []
        rejected_end_scores = []

        for i in range(bs):
            # Get last non-padding token position for chosen sequence
            c_mask = decoder_attention_mask[i].bool()
            c_last_idx = c_mask.nonzero()[-1] if c_mask.any() else 0
            chosen_end_scores.append(chosen_rewards[i, c_last_idx])

            # Get last non-padding token position for rejected sequence
            r_mask = decoder_attention_mask[bs + i].bool()
            r_last_idx = r_mask.nonzero()[-1] if r_mask.any() else 0
            rejected_end_scores.append(rejected_rewards[i, r_last_idx])

        chosen_end_scores = torch.stack(chosen_end_scores)
        rejected_end_scores = torch.stack(rejected_end_scores)

        # Compute loss
        loss = -torch.log(torch.sigmoid(chosen_end_scores - rejected_end_scores)).mean()

        return loss, chosen_end_scores, rejected_end_scores


# In[4]:


import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, Trainer, TrainingArguments


def create_comparison_dataset(path):
    dataset = pd.read_csv(path)
    pairs = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing preferences"):
        pair = {}
        prompt = row["findings"]
        chosen_summary = row["impression"]
        rejected_summary = row["t5_summary"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        # Format for T5: prefix the input with "summarize: "
        pair["chosen_input"] = "summarize: " + prompt
        pair["chosen_output"] = chosen_summary
        pair["rejected_input"] = "summarize: " + prompt
        pair["rejected_output"] = rejected_summary
        pairs.append(pair)
    return pairs


# In[5]:


class PairwiseDatasetT5(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attention_masks = []
        self.chosen_decoder_input_ids = []
        self.chosen_decoder_attention_masks = []
        self.rejected_input_ids = []
        self.rejected_attention_masks = []
        self.rejected_decoder_input_ids = []
        self.rejected_decoder_attention_masks = []

        for pair in tqdm(pairs):
            # tokenize inputs (prompts)
            chosen_input_encodings = tokenizer(
                pair["chosen_input"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            # tokenize outputs (summaries)
            chosen_output_encodings = tokenizer(
                pair["chosen_output"],
                truncation=True,
                max_length=max_length // 2,  # shorter for summaries
                padding="max_length",
                return_tensors="pt",
            )

            rejected_input_encodings = tokenizer(
                pair["rejected_input"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            rejected_output_encodings = tokenizer(
                pair["rejected_output"],
                truncation=True,
                max_length=max_length // 2,
                padding="max_length",
                return_tensors="pt",
            )

            # decoder inputs (shift right, add starting token)
            chosen_decoder_input_ids = self.prepare_decoder_input_ids(chosen_output_encodings["input_ids"], tokenizer)
            rejected_decoder_input_ids = self.prepare_decoder_input_ids(rejected_output_encodings["input_ids"], tokenizer)

            # only add if the chosen and rejected outputs are different
            if not torch.all(torch.eq(chosen_decoder_input_ids, rejected_decoder_input_ids)).item():
                self.chosen_input_ids.append(chosen_input_encodings["input_ids"])
                self.chosen_attention_masks.append(chosen_input_encodings["attention_mask"])
                self.chosen_decoder_input_ids.append(chosen_decoder_input_ids)
                self.chosen_decoder_attention_masks.append(chosen_output_encodings["attention_mask"])
                
                self.rejected_input_ids.append(rejected_input_encodings["input_ids"])
                self.rejected_attention_masks.append(rejected_input_encodings["attention_mask"])
                self.rejected_decoder_input_ids.append(rejected_decoder_input_ids)
                self.rejected_decoder_attention_masks.append(rejected_output_encodings["attention_mask"])

    def prepare_decoder_input_ids(self, labels, tokenizer):
        # shift the labels to create decoder_input_ids
        decoder_input_ids = labels.clone()
        decoder_input_ids = torch.cat([
            torch.tensor([[tokenizer.pad_token_id]]), 
            decoder_input_ids[:, :-1]], dim=1)
        return decoder_input_ids

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attention_masks[idx],
            self.chosen_decoder_input_ids[idx],
            self.chosen_decoder_attention_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attention_masks[idx],
            self.rejected_decoder_input_ids[idx],
            self.rejected_decoder_attention_masks[idx],
        )


# In[6]:


class DataCollatorRewardT5:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[4] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[5] for f in data])
        batch["decoder_input_ids"] = torch.cat([f[2] for f in data] + [f[6] for f in data])
        batch["decoder_attention_mask"] = torch.cat([f[3] for f in data] + [f[7] for f in data])
        return batch


# In[7]:


def compute_metrics(eval_preds):
    _, chosen_end_scores, rejected_end_scores = eval_preds.predictions
    
    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc
    
    return result


# In[8]:


tokenizer = T5Tokenizer.from_pretrained('t5-base')


# In[9]:


if not os.path.exists("../models/reward_model_t5"):
    os.mkdir("../models/reward_model_t5")

training_args = TrainingArguments(
    output_dir="../models/reward_model_t5/",
    num_train_epochs=5,
    logging_steps=10,
    gradient_accumulation_steps=4,
    save_strategy="no",
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
#     eval_steps=100,
#     save_steps=0,
#     load_best_model_at_end=True,
    warmup_steps=100,
    logging_dir="../logs/reward_model_t5",
    log_level='info',
    report_to="tensorboard",
    fp16=True,
    bf16=False,
    learning_rate=1e-5,
    save_total_limit=1,
    prediction_loss_only=False,
)


# In[10]:


# initialize reward model from the fine-tuned T5
model = T5RewardModel("../models/sft_t5/checkpoint-730")

# freeze first 70% of the encoder and decoder layers
encoder_layers = model.encoder.block
decoder_layers = model.decoder.block
num_encoder_layers = len(encoder_layers)
num_decoder_layers = len(decoder_layers)
num_unfrozen_encoder = int(0.3 * num_encoder_layers)
num_unfrozen_decoder = int(0.3 * num_decoder_layers)

# freeze encoder layers
for layer in encoder_layers[:-num_unfrozen_encoder]:
    layer.requires_grad_(False)

# freeze decoder layers
for layer in decoder_layers[:-num_unfrozen_decoder]:
    layer.requires_grad_(False)


# In[11]:


# preference datasets
train_pairs = create_comparison_dataset("../data/preference_dataset/preference_train.csv")
val_pairs = create_comparison_dataset("../data/preference_dataset/preference_val.csv")

# pairwise datasets for training
max_length = 512
train_dataset = PairwiseDatasetT5(train_pairs, tokenizer, max_length=max_length)
val_dataset = PairwiseDatasetT5(val_pairs, tokenizer, max_length=max_length)

# collator to gather batches of pairwise comparisons
data_collator = DataCollatorRewardT5()


# In[ ]:


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

trainer.train()


# In[ ]:


from safetensors.torch import load_model, save_model

save_model(trainer.model, "../models/reward_model_t5/reward_model_t5.safetensors")