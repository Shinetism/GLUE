import numpy as np
import pandas as pd
import torch
import wget, os
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

df = pd.read_csv(
    '.\\cola_public\\raw\\in_domain_train.tsv',
    delimiter='\t', header=None,
    names=['sentence_source', 'label', 'label_notes', 'sentence']
)
# print('Number of training sentences: {:,}\n'.format(df.shape[0]))
sentences = df.sentence.values
labels = df.label.values

# Tokenize Datasets
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
input_ids = []
attention_masks = []
for sent in sentences:
    encode_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encode_dict['input_ids'])
    attention_masks.append(encode_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# print('Original: ', sentences[0])
# print('Token IDs:', input_ids[0])

# DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataset = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Training Model
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
model.to(device)

## Optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5, eps=1e-8)


model.train()

