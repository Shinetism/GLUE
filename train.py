import numpy as np
import pandas as pd
import torch
import random
import time
import datetime
import wget, os
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup

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

## Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr = 2e-5, eps=1e-8)
epoch = 4
total_steps = len(train_dataloader) * epoch
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)

# Function to format time
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set random seeds
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

training_stats = []

total_t0 = time.time()

for epoch_i in range(epoch):
    # Training the model
    print()
    print('========== EPOCH {:}/{:} ============'.format(epoch_i+1, epoch))
    print('Training')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if not step == 0:
            elapsed = format_time(time.time()-t0)
            print(' Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        model.zero_grad()
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels).to_tuple()
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss/len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print()
    print(' Average training loss: {0:.2f}'.format(avg_train_loss))
    print(' Training epoch took: {:}'.format(training_time))
