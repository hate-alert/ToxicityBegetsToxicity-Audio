import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bdotloh/distilbert-base-uncased-empathetic-dialogues-context", clean_up_tokenization_spaces=True)
model = AutoModelForSequenceClassification.from_pretrained("bdotloh/distilbert-base-uncased-empathetic-dialogues-context").to("cuda:0")

def classifier(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = (-logits).argsort()[0][:3]
    arr_item = []
    for item in logits:
        arr_item.append(model.config.id2label[item.item()])
    return arr_item

with open('/kaggle/input/chain-final/updated_conversations_with_empath.json', 'r') as data_string:
    data = json.load(data_string)

for chain in tqdm(data["chains"]):
    prev_text = ""
    for element in chain["Before Conversations"]:
        arr = classifier(element["text"])
        element["Empath"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = classifier(joined_text)
        element["Concatenated_Empath"] = arr
        prev_text = element["text"]

    for element in chain["Toxic Conversation"]:
        arr = classifier(element["text"])
        element["Empath"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = classifier(joined_text)
        element["Concatenated_Empath"] = arr
        prev_text = element["text"]

    for element in chain["After Conversations"]:
        arr = classifier(element["text"])
        element["Empath"] = arr
        if prev_text == "":
            prev_text = element["text"]
            continue
        joined_text = prev_text + " " + element["text"]
        arr = classifier(joined_text)
        element["Concatenated_Empath"] = arr
        prev_text = element["text"]

with open('updated_conversations_with_empath.json', 'w') as fp:
    json.dump(data, fp, indent=6)
fp.close()