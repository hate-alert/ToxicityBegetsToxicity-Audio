import json
import os
import numpy as np

import tqdm
from googleapiclient.errors import HttpError
from langdetect import detect, LangDetectException

from core_classes import JSONParser, CommentAnalyzer

json_parser = JSONParser()

complete_output = {"files": []}

total_segments = 0
false_counter = 0
true_counter = 0

API_KEY = ''

analyzer = CommentAnalyzer(API_KEY)

def divide_text(text, number_of_parts):
    words = text.split()
    if number_of_parts < 1:
        number_of_parts = 1
    k, m = divmod(len(words), number_of_parts)
    return [" ".join(words[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(number_of_parts)]


def perspective_api_chunked(file_path):
    # Load the original JSON data
    with open(file_path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)
    
    segments = data_dict.get("conversations", [])

    # Process each segment in the conversations
    for segment in segments:
        conversation = segment.get("conversation", [])
        for index, seg in enumerate(conversation):
            text = seg.get("text", "").strip()
            if text:
                try:
                    if detect(text) == 'en':
                        duration = seg["end_time"] - seg["start_time"]
                        parts = int(duration / 17)
                        divided_texts = divide_text(text, parts)
                        
                        toxicity_scores = []
                        insult_scores = []
                        threat_scores = []
                        profanity_scores = []
                        
                        for part_text in divided_texts:
                            response = analyzer.analyze_text(part_text)
                            if response.get("languages", []) == ['en']:
                                toxicity = response["attributeScores"]['TOXICITY']['summaryScore']['value']
                                insult = response["attributeScores"]['INSULT']['summaryScore']['value']
                                threat = response["attributeScores"]['THREAT']['summaryScore']['value']
                                profanity = response["attributeScores"]['PROFANITY']['summaryScore']['value']
                                
                                toxicity_scores.append(toxicity)
                                insult_scores.append(insult)
                                threat_scores.append(threat)
                                profanity_scores.append(profanity)
                        
                        seg["TOXICITY"] = toxicity_scores
                        seg["INSULT"] = insult_scores
                        seg["THREAT"] = threat_scores
                        seg["PROFANITY"] = profanity_scores
                    
                    else:
                        seg["TOXICITY"] = []
                        seg["INSULT"] = []
                        seg["THREAT"] = []
                        seg["PROFANITY"] = []
                        
                except (LangDetectException, HttpError) as e:
                    print(f"An error occurred: {e}")
                    seg["TOXICITY"] = []
                    seg["INSULT"] = []
                    seg["THREAT"] = []
                    seg["PROFANITY"] = []
    
    # Update the 'conversations' key with the modified segments
    data_dict["conversations"] = segments
    
    # Write the updated data back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, indent=4)


def is_any_array_element_above_threshold(generic_array, threshold):
    for item in generic_array:
        if item >= threshold:
            return True
    return False


def count_most_toxic(file_path, segments):
    global total_segments
    global false_counter
    global true_counter

    toxicity_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    insult_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    profanity_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    threat_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    assert len(toxicity_count) == len(insult_count) == len(profanity_count) == len(threat_count) == 10
    assert len(levels) == 10

    # Assuming segments is a list of dictionaries
    for segment in segments:
        conversation = segment["conversation"]
        for seg in conversation:
            total_segments += 1
            if "TOXICITY" not in seg:
                assert (("INSULT" not in seg) and ("PROFANITY" not in seg) and ("THREAT" not in seg))
                false_counter += 1
            else:
                assert (("INSULT" in seg) and ("PROFANITY" in seg) and ("THREAT" in seg))
                true_counter += 1
                for iterator, level in enumerate(levels):
                    assert iterator == level * 10

                    if "TOXICITY" in seg and is_any_array_element_above_threshold(seg["TOXICITY"], level):
                        toxicity_count[iterator] += 1

                    if "INSULT" in seg and is_any_array_element_above_threshold(seg["INSULT"], level):
                        insult_count[iterator] += 1

                    if "PROFANITY" in seg and is_any_array_element_above_threshold(seg["PROFANITY"], level):
                        profanity_count[iterator] += 1

                    if "THREAT" in seg and is_any_array_element_above_threshold(seg["THREAT"], level):
                        threat_count[iterator] += 1

        fil_seg = {
            "file_path": file_path,
            "TOXICITY": toxicity_count,
            "INSULT": insult_count,
            "PROFANITY": profanity_count,
            "THREAT": threat_count
        }

        complete_output["files"].append(fil_seg)


def calculate_podcast_statistics(file_path_stats, segments_stats, hash_map_plus):
    podcast_channel = file_path_stats.split("/")[-2].replace("_", " ")

    if podcast_channel == "PragerU Five Minute Videos":
        return

    if podcast_channel not in hash_map_plus:
        hash_map_plus[podcast_channel] = {
            "counter": 0,
            "toxic_episodes": 0,
            "duration": [],
            "word_count": []
        }

    hash_map_plus[podcast_channel]["counter"] += 1
    duration_ = 0
    words_ = 0
    toxicity_flag_ = 0

    for segment in segments_stats:
        conversation = segment["conversation"]
        for seg in conversation:
            duration_ += (seg["end_time"] - seg["start_time"])
            words_ += seg["word_count"]
            if "TOXICITY" in seg and len(seg["TOXICITY"]) > 0 and max(seg["TOXICITY"]) >= 0.7:
                toxicity_flag_ = 1

    hash_map_plus[podcast_channel]["duration"].append(duration_)
    hash_map_plus[podcast_channel]["word_count"].append(words_)
    hash_map_plus[podcast_channel]["toxic_episodes"] += toxicity_flag_


# Specify the path to the JSON file
json_folder_path = "../diarize_with_perpective_api_detoxify"

data_of_each_file = json_parser.extract_information_from_folder(json_folder_path)

hash_map = {

}

for segments, file_path in tqdm.tqdm(data_of_each_file):
    # print(file_path)

    # Stats
    calculate_podcast_statistics(file_path, segments, hash_map)

    # Helper to arrange and get most toxic files. The output of this function will later help to build toxic
    # conversation chains
    # count_most_toxic(file_path, segments)

    # pers_api _code_chunked
    # perspective_api_chunked(file_path)

# Please uncomment the code which you want to run

for item in hash_map:
    print(item)
    # print(hash_map[item]["counter"])

    # duration_average = np.mean(hash_map[item]["duration"])
    # duration_std_dev = np.std(hash_map[item]["duration"])
    # print(round(duration_average / 60), " (", round(duration_std_dev / 60), ")", sep='')

    # words_average = np.mean(hash_map[item]["word_count"])
    # words_std_dev = np.std(hash_map[item]["word_count"])
    # print(round(words_average), " (", round(words_std_dev), ")", sep='')

    # print(hash_map[item]["toxic_episodes"])

    # print(round((hash_map[item]["toxic_episodes"] * 100) / hash_map[item]["counter"]))

# print("Total_segments :", total_segments)
# print("False Counter:", false_counter, "/ True Counter:", true_counter)

# output_file = open(os.path.join("../result", "perspective_api_analysis_17_sec.json"), "w")
# json_object = json.dumps(complete_output, indent=4)
# output_file.write(json_object)
# output_file.close()
