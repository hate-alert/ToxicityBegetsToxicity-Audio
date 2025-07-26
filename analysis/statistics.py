import json

with open("path.json", "r") as file:
    data = json.load(file)

data = data["chains"]

total_chains = len(data)

hashmap = {}

for item in data:
    file_name = item["Podcast File Name"]
    set_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "-", ".json"]
    for itemize in set_:
        file_name = file_name.replace(str(itemize), "")
    file_name = file_name.replace("_", " ")

    if file_name not in hashmap:
        hashmap[file_name] = 0
    hashmap[file_name] += 1

for itemize in hashmap:
    # print(itemize.strip())
    print(itemize.strip(), round((hashmap[itemize] * 100) / total_chains, 1))


