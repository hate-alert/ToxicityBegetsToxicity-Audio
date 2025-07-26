model_name = "bert-large-uncased"
seed_val = 42
# Change the numbers in random_list for 3 runs of 10-fold cross-validation run
random_list = [42,101,2020]
# Change random_st variable for the 10-fold cross-validation run
random_st = 42
# Hyperparameters
MAX_LEN=64
batch_size = 16
epochs = 10
learning_rate = 1e-5

from transformers import *
import random
import pandas as pd
import numpy as np
import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from tqdm import tqdm, trange
import tensorflow as tf
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

# Set the seed value
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# Check for GPU device
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found!')
print('GPU found at: {}'.format(device_name))
# Set the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('/kaggle/input/hypo-dataset/hypo-l.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna(axis=0).reset_index(drop=True)
df["Hyperbole"] = df["Hyperbole"].astype("int")
df["Metaphor"] = df["Metaphor"].astype("int")
cols = df.columns
label_cols = list(cols[1:])
num_labels = len(label_cols)
df['one_hot_labels'] = list(df[label_cols].values)
conditions = [
    (df["Hyperbole"]==0) & (df["Metaphor"]==0),
    (df["Hyperbole"]==0) & (df["Metaphor"]==1),
    (df["Hyperbole"]==1) & (df["Metaphor"]==0),
    (df["Hyperbole"]==1) & (df["Metaphor"]==1)
]
choices = [0,1,2,3]

df["new"] = np.select(conditions, choices)
y= df["new"].values
labels = list(df.one_hot_labels.values)
comments = list(df.Sentence.values)

model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

for i in range(len(df)):
    tokenizer.padding_side = 'right'
    try:
        encoded_sent = tokenizer.encode_plus(
                            df['Sentence'][i],            # Sentence to encode.
                            add_special_tokens = True,
                            max_length = MAX_LEN,
                            pad_to_max_length = True
                            )
    except Exception as e:
        print(e)

    input_ids.append(encoded_sent['input_ids'])
    attention_masks.append(encoded_sent['attention_mask'])

final_list = []
for ran in random_list:
    random_state = ran
    kf = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)

    f1_list = []
    for i, (train_index, validation_index) in enumerate(kf.split(input_ids, y)):
        # if i==2:
        #   break
        z = i
        print()
        print("---------------------------------FOLD NO: ", i)
        print()
        train_inputs = list(itemgetter(*train_index)(input_ids))
        train_labels = list(itemgetter(*train_index)(labels))
        train_masks = list(itemgetter(*train_index)(attention_masks))

        validation_inputs = list(itemgetter(*validation_index)(input_ids))
        validation_labels = list(itemgetter(*validation_index)(labels))
        validation_masks = list(itemgetter(*validation_index)(attention_masks))

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                   output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model.cuda()

        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          eps=1e-8
                          )

        train_loss_set = []

        print('len(train_dataloader)', len(train_dataloader))
        total_steps = len(train_dataloader) * epochs
        print('total_steps', total_steps)
        warmup_steps = int(0.06 * total_steps)
        print('warmup_steps', warmup_steps)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        for _ in trange(epochs, desc="Epoch"):
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                optimizer.zero_grad()

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]
                loss_func = BCEWithLogitsLoss()
                loss = loss_func(logits.view(-1, num_labels), b_labels.type_as(logits).view(-1,
                                                                                            num_labels))  # convert labels to float for calculation
                train_loss_set.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            ###############################################################################

            # Validation
            model.eval()

            logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

            for i, batch in enumerate(validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    b_logit_pred = outs[0]
                    pred_label = torch.sigmoid(b_logit_pred)

                    b_logit_pred = b_logit_pred.detach().cpu().numpy()
                    pred_label = pred_label.to('cpu').numpy()
                    b_labels = b_labels.to('cpu').numpy()

                tokenized_texts.append(b_input_ids)
                logit_preds.append(b_logit_pred)
                true_labels.append(b_labels)
                pred_labels.append(pred_label)

            pred_labels = [item for sublist in pred_labels for item in sublist]
            true_labels = [item for sublist in true_labels for item in sublist]

            threshold = 0.50
            pred_bools = [pl > threshold for pl in pred_labels]
            true_bools = [tl == 1 for tl in true_labels]
            val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
            val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

            print('F1 Validation Accuracy: ', val_f1_accuracy)
            print('Flat Validation Accuracy: ', val_flat_accuracy)

        model.eval()

        logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outs = model(b_input_ids, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        true_bools = [tl == 1 for tl in true_labels]

        pred_bools = [pl > 0.50 for pl in pred_labels]

        # Print and save classification report
        print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools, average='micro'))
        print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools), '\n')
        clf_report = classification_report(true_bools, pred_bools, target_names=label_cols)

        print(clf_report)
        f1_list.append(clf_report)

    final_list.extend(f1_list)

def hyperbole(text):
    inputs = tokenizer.encode(text, truncation=True, max_length=512, return_tensors='pt').to('cuda:1')
    out = model(inputs.to('cuda:1'))
    b_logit_pred = out[0]
    pred_label = torch.sigmoid(b_logit_pred)
    pred_bools = [pl>0.50 for pl in pred_label]
    return pred_bools[0][0].cpu().numpy()

with open('/kaggle/input/metaphor/toxic_metaphor.json', 'r') as data_string:
    data = json.load(data_string)

for chain in tqdm(data["chains"]):
    prev_text = ""
    for element in chain["Before Conversations"]:
        arr = hyperbole(element["text"])
        element["Hyperbole"] = bool(arr)
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = hyperbole(joined_text)
        element["Concatenated_hyperbole"] = bool(arr)
        prev_text = element["text"]

    for element in chain["Toxic Conversation"]:
        arr = hyperbole(element["text"])
        element["Hyperbole"] = bool(arr)
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = hyperbole(joined_text)
        element["Concatenated_hyperbole"] = bool(arr)
        prev_text = element["text"]

    for element in chain["After Conversations"]:
        arr = hyperbole(element["text"])
        element["Hyperbole"] = bool(arr)
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = hyperbole(joined_text)
        element["Concatenated_hyperbole"] = bool(arr)
        prev_text = element["text"]

with open("toxic_final.json", "w") as outfile:
    json.dump(data, outfile)