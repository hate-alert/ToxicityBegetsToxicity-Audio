import json
from tqdm import tqdm

from core_classes import CommentAnalyzer

API_KEY = ''
analyzer = CommentAnalyzer(API_KEY)

def get_toxicity(text):
    response = analyzer.analyze_text(text)
    if response["languages"] == ['en']:
        return response["attributeScores"]['TOXICITY']['summaryScore']['value']

# Process a single chain
def process_chain(chain_element):
    print(chain_element["Podcast File Name"])
    counter = 0
    prev_text = ""

    for itemize in chain_element["Before Conversations"]:
        counter += 1
        if counter == 1:
            prev_text = itemize["text"]
            continue
        itemize['Concatenated_Toxicity'] = get_toxicity(prev_text + " " + itemize["text"])
        prev_text = itemize["text"]

    for itemize in chain_element["Toxic Conversation"]:
        counter += 1
        if counter == 1:
            prev_text = itemize["text"]
            continue
        itemize['Concatenated_Toxicity'] = get_toxicity(prev_text + " " + itemize["text"])
        prev_text = itemize["text"]

    for itemize in chain_element["After Conversations"]:
        assert counter >= 1
        itemize['Concatenated_Toxicity'] = get_toxicity(prev_text + " " + itemize["text"])
        prev_text = itemize["text"]

input_file = "top_toxic_chains_liberals.json"
output_file = "top_toxic_chains_liberals_concat.json"

with open(input_file, "r") as f:
    data = json.load(f)

for chain in tqdm(data):
    process_chain(chain)

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)
