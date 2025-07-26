import torch
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lenguist/longformer-coherence-synthetic-classifier")
model = AutoModelForSequenceClassification.from_pretrained("lenguist/longformer-coherence-synthetic-classifier").to("cuda:0")

def coherence_score(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

with open('/kaggle/input/chain-final/updated_conversations_with_empath.json', 'r') as data_string:
    data = json.load(data_string)

for chain in tqdm(data["chains"]):
    prev_text = ""
    for element in chain["Before Conversations"]:
        arr = coherence_score(element["text"])
        element["Coherence"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = coherence_score(joined_text)
        element["Concatenated_Coherence"] = arr
        prev_text = element["text"]

    for element in chain["Toxic Conversation"]:
        arr = coherence_score(element["text"])
        element["Coherence"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = coherence_score(joined_text)
        element["Concatenated_Coherence"] = arr
        prev_text = element["text"]

    for element in chain["After Conversations"]:
        arr = coherence_score(element["text"])
        element["Coherence"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = coherence_score(joined_text)
        element["Concatenated_Coherence"] = arr
        prev_text = element["text"]

with open('updated_conversations_with_empath.json', 'w') as fp:
    json.dump(data, fp, indent=6)
fp.close()