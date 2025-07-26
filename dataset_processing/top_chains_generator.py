from tqdm import tqdm
import json

with open("toxic_conversation_chains_path.json", "r") as file:
    data = json.load(file)

hash_map = {}

for item in tqdm(data["chains"]):
    toxicity_score = max(item["Toxic Conversation"][0]["TOXICITY"])
    if -toxicity_score not in hash_map:
        hash_map[-toxicity_score] = []
    hash_map[-toxicity_score].append(item["Toxic Chain Index"])


sorted_hash_map = sorted(hash_map.items())

counter = []

for toxicity, index_vector in sorted_hash_map:
    for index in index_vector:
        if index not in [664, 2830, 665, 398]:
            counter.append(index)
        if len(counter) == 100:
            break
    if len(counter) == 100:
        break

top_chains = []
for index in counter:
    for item in tqdm(data["chains"]):
        if item["Toxic Chain Index"] == index:
            top_chains.append(item)

with open(f'top_toxic_chains_liberals.json', 'w') as fp:
    json.dump(top_chains, fp, indent=5)
