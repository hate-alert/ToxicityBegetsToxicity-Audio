import json

import re
from tqdm import tqdm

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

def save_file_key_words():
    with open("/Users/naquee.rizwan/Desktop/fear-speech/podcast/toxic_conversation_chains_band_0.7_to_1"
              ".0_sampled_at_60_seconds.json", "r") as data_file:
        file_contents = json.load(data_file)

    key_bert_phrases_map = {
        "-10": { "key_bert_phrases": [] },
        "-9": { "key_bert_phrases": [] },
        "-8": { "key_bert_phrases": [] },
        "-7": { "key_bert_phrases": [] },
        "-6": { "key_bert_phrases": [] },
        "-5": { "key_bert_phrases": [] },
        "-4": { "key_bert_phrases": [] },
        "-3": { "key_bert_phrases": [] },
        "-2": { "key_bert_phrases": [] },
        "-1": { "key_bert_phrases": [] },
        "0": { "key_bert_phrases": [] },
        "1": { "key_bert_phrases": [] },
        "2": { "key_bert_phrases": [] },
        "3": { "key_bert_phrases": [] },
        "4": { "key_bert_phrases": [] },
        "5": { "key_bert_phrases": [] },
        "6": { "key_bert_phrases": [] },
        "7": { "key_bert_phrases": [] },
        "8": { "key_bert_phrases": [] },
        "9": { "key_bert_phrases": [] },
        "10": { "key_bert_phrases": [] }
    }

    for chain in tqdm(file_contents["chains"]):
        before_conversation_length = len(chain["Before Conversations"])

        for index, element in enumerate(chain["Before Conversations"]):
            relative_index = "-" + str(before_conversation_length - index)

            for keyword in element["Keywords"]:
                key_bert_phrases_map[relative_index]["key_bert_phrases"].append(keyword[0])

        for index, element in enumerate(chain["Toxic Conversation"]):
            assert index == 0

            for keyword in element["Keywords"]:
                key_bert_phrases_map[str(index)]["key_bert_phrases"].append(keyword[0])

        for index, element in enumerate(chain["After Conversations"]):
            relative_index = str(1 + index)

            for keyword in element["Keywords"]:
                key_bert_phrases_map[relative_index]["key_bert_phrases"].append(keyword[0])

    with open(f"/Users/naquee.rizwan/Desktop/fear-speech/podcast/key_phrase_map_sorted.json",
              "w") as key_phrase_map_file:
        json.dump(key_bert_phrases_map, key_phrase_map_file, indent=4)


def generate_and_save_word_clouds(text, file_name):
    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    plt.savefig(f'{file_name}.pdf', format='pdf')

    plt.show()
    plt.close()

def word_cloud():
    with open("/Users/naquee.rizwan/Desktop/fear-speech/podcast/key_phrase_map_sorted.json", "r") as data_file:
        file_contents = json.load(data_file)

    previous_phrases = []
    anchor_phrases = []
    next_phrases = []

    for item in file_contents:
        if int(item) < 0:
            previous_phrases.extend(file_contents[item]["key_bert_phrases"])
        elif int(item) == 0:
            anchor_phrases.extend(file_contents[item]["key_bert_phrases"])
        else:
            assert 1 <= int(item) <= 10
            next_phrases.extend(file_contents[item]["key_bert_phrases"])

    print(len(previous_phrases), len(anchor_phrases), len(next_phrases))
    previous_phrases = ' '.join(previous_phrases)
    anchor_phrases = ' '.join(anchor_phrases)
    next_phrases = ' '.join(next_phrases)

    previous_phrases = re.sub(r'[^\w\s]', '', previous_phrases.lower())
    anchor_phrases = re.sub(r'[^\w\s]', '', anchor_phrases.lower())
    next_phrases = re.sub(r'[^\w\s]', '', next_phrases.lower())

    print(len(previous_phrases), len(anchor_phrases), len(next_phrases))

    stop_words = set(stopwords.words('english'))
    previous_phrases = ' '.join(word for word in previous_phrases.split() if word not in stop_words)
    anchor_phrases = ' '.join(word for word in anchor_phrases.split() if word not in stop_words)
    next_phrases = ' '.join(word for word in next_phrases.split() if word not in stop_words)

    print(len(previous_phrases), len(anchor_phrases), len(next_phrases))

    generate_and_save_word_clouds(previous_phrases, "previous_phrases")
    generate_and_save_word_clouds(anchor_phrases, "anchor_phrases")
    generate_and_save_word_clouds(next_phrases, "next_phrases")

# save_file_key_words()
word_cloud()
