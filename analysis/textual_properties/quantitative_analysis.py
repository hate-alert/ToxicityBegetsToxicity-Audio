import json
from tqdm import tqdm
from nltk import word_tokenize
import os
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

from pynlpl.statistics import Distribution, FrequencyList

array_for_empath = ["surprised", "angry", "hopeful", "afraid", "anticipating", "furious", "apprehensive", "trusting", "annoyed", "impressed", "disgusted", "confident", "faithful", "jealous", "embarrassed", "grateful", "anxious", "terrified", "nostalgic", "excited", "disappointed", "content", "joyful", "devastated", "ashamed", "prepared", "sad", "proud", "sentimental", "guilty", "lonely", "caring"]
band = [2, 3, 4]
seed = [42, 101, 2020]

band_index = 0
seed_index = 0

def confidence_interval(data):
    # Calculate mean and standard error of the mean
    mean = np.mean(data)
    sem = stats.sem(data)

    # Define confidence level and calculate the confidence interval
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    confidence_interval_ = stats.t.interval(confidence_level, degrees_freedom, mean, sem)

    return confidence_interval_

def word_statistics(text):
    words = word_tokenize(text)

    freq_list = FrequencyList(words)
    word_counts = dict(freq_list.items())

    distribution = Distribution(word_counts)

    return {
        "token_statistics": freq_list.tokens(),
        "ttr": freq_list.typetokenratio(),
        "entropy": distribution.entropy(),
        "perplexity": distribution.perplexity()
    }

def compute_values(chain_wise_map, rel_index, elem):
    text_statistics = word_statistics(elem["text"])

    assert len(elem["Empath"]) == 3

    chain_wise_map[rel_index]["empath_values"].append(elem["Empath"])

    chain_wise_map[rel_index]["ttr"].append(round(text_statistics["ttr"], 2))
    chain_wise_map[rel_index]["time_coverage"].append(round(elem["end_time"] - elem["start_time"], 2))
    chain_wise_map[rel_index]["token_statistics"].append(text_statistics["token_statistics"])

    chain_wise_map[rel_index]["hyperbole"].append(elem["Hyperbole"])
    chain_wise_map[rel_index]["metaphor"].append(elem["Metaphor"])

    chain_wise_map[rel_index]["entropy"].append(round(text_statistics["entropy"], 2))
    chain_wise_map[rel_index]["perplexity"].append(round(text_statistics["perplexity"], 2))

def store_files(
        file_path="toxic_conversation_chains_band_0.7_to_1.0_sampled_at_60_seconds.json",
        output_file_path="chain_wise_map.json"
):
    with (open(os.path.join('/Users/naquee.rizwan/Desktop/fear-speech/podcast', file_path), 'r')) as data_string:
        data = json.load(data_string)

    chain_wise_map = {
        "-10": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-9": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-8": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-7": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-6": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-5": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-4": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-3": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-2": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-1": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "0": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "1": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "2": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "3": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "4": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "5": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "6": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "7": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "8": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "9": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "10": { "empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
    }

    for chain in tqdm(data["chains"]):

        before_conversation_length = len(chain["Before Conversations"])

        for index, element in enumerate(chain["Before Conversations"]):
            relative_index = "-" + str(before_conversation_length - index)
            compute_values(chain_wise_map, relative_index, element)

        for index, element in enumerate(chain["Toxic Conversation"]):
            assert index == 0
            compute_values(chain_wise_map, str(index), element)

        for index, element in enumerate(chain["After Conversations"]):
            relative_index = str(1 + index)
            compute_values(chain_wise_map, relative_index, element)

    with open(f"/Users/naquee.rizwan/Desktop/fear-speech/podcast/{output_file_path}", "w") as chain_wise_map_file:
        json.dump(chain_wise_map, chain_wise_map_file, indent=4)

def gather_statistics(y_axis, printable_metric, input_file="chain_wise_map.json", prefix=""):
    chain_wise_map_overall = {
        "-10": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-9": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-8": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-7": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-6": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-5": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-4": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-3": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-2": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "-1": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "0": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "1": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "2": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "3": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "4": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "5": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "6": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "7": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "8": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "9": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
        "10": {"empath_values": [], "ttr": [], "time_coverage": [], "token_statistics": [], "hyperbole": [], "metaphor": [], "entropy": [], "perplexity": [] },
    }

    for seed_individual in seed[:1]:
        with open(f"/Users/naquee.rizwan/Desktop/fear-speech/podcast/{input_file}","r") as chain_wise_map_file:
            chain_wise_map = json.load(chain_wise_map_file)

        for item in chain_wise_map:
            chain_wise_map_overall[item]["ttr"].append(
                (round(np.mean(chain_wise_map[item]["ttr"]), 2),
                 round(np.std(chain_wise_map[item]["ttr"]), 2),
                 confidence_interval(chain_wise_map[item]["ttr"]))
            )

            chain_wise_map_overall[item]["time_coverage"].append(
                (round(np.mean(chain_wise_map[item]["time_coverage"]), 2),
                 round(np.std(chain_wise_map[item]["time_coverage"]), 2),
                 confidence_interval(chain_wise_map[item]["time_coverage"]))
            )

            chain_wise_map_overall[item]["token_statistics"].append(
                (round(np.mean(chain_wise_map[item]["token_statistics"]), 2),
                 round(np.std(chain_wise_map[item]["token_statistics"]), 2),
                 confidence_interval(chain_wise_map[item]["token_statistics"]))
            )

            chain_wise_map_overall[item]["entropy"].append(
                (round(np.mean(chain_wise_map[item]["entropy"]), 2),
                 round(np.std(chain_wise_map[item]["entropy"]), 2),
                 confidence_interval(chain_wise_map[item]["entropy"]))
            )

            chain_wise_map_overall[item]["perplexity"].append(
                (round(np.mean(chain_wise_map[item]["perplexity"]), 2),
                 round(np.std(chain_wise_map[item]["perplexity"]), 2),
                 confidence_interval(chain_wise_map[item]["perplexity"]))
            )

            map_emotion = {}

            for item_emotion in chain_wise_map[item]["empath_values"]:
                if item_emotion[0] not in map_emotion:
                    map_emotion[item_emotion[0]] = 0
                map_emotion[item_emotion[0]] += 1

            sorted_empath = dict(sorted(map_emotion.items(), key=lambda item_emot: -item_emot[1]))

            total_sum = 0
            for emotion in sorted_empath:
                total_sum += sorted_empath[emotion]

            for emotion in sorted_empath:
                sorted_empath[emotion] = round((sorted_empath[emotion] / total_sum) * 100, 2)

            percentage_sum = 0
            assert len(sorted_empath) == len(array_for_empath)
            for emotion in sorted_empath:
                percentage_sum += sorted_empath[emotion]
                assert emotion in array_for_empath
            assert round(percentage_sum) == 100

            counter_hyperbole = 0
            for item_hyperbole in chain_wise_map[item]["hyperbole"]:
                if item_hyperbole == 1:
                    counter_hyperbole += 1
            chain_wise_map_overall[item]["hyperbole"].append(
                round(100 * (counter_hyperbole / len(chain_wise_map[item]["hyperbole"])), 2)
            )
            # print(round(100 * (counter_hyperbole / len(chain_wise_map[item]["hyperbole"])), 2))

            counter_metaphor = 0
            for item_metaphor in chain_wise_map[item]["metaphor"]:
                if item_metaphor == 1:
                    counter_metaphor += 1
            chain_wise_map_overall[item]["metaphor"].append(
                round(100 * (counter_metaphor / len(chain_wise_map[item]["metaphor"])), 2)
            )
            # print(round(100 * (counter_metaphor / len(chain_wise_map[item]["metaphor"])), 2))

    mean_overall = []
    confidence_overall = []

    for item in chain_wise_map_overall:
        assert len(chain_wise_map_overall[item][printable_metric]) == 1
        mean_chain, std_dev_chain, confidence = [], [], []
        for metric in chain_wise_map_overall[item][printable_metric]:
            m, s, c = metric
            mean_chain.append(m)
            std_dev_chain.append(s)
            confidence.append(c)

            mean_overall.append(m)
            confidence_overall.append(c)

    lower_overall = []
    upper_overall = []
    for index, iiii in enumerate(confidence_overall):
        lower_overall.append(max(0, mean_overall[index] - iiii[0]))
        upper_overall.append(max(0, iiii[1] - mean_overall[index]))

    array_indices = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    array_labels = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(array_indices, mean_overall, marker='o', color='#1f77b4', linestyle='-', linewidth=3, markersize=8)
    ax.errorbar(array_indices, mean_overall, yerr=[lower_overall, upper_overall], fmt='o', color='#1f77b4',
                linewidth=3, capsize=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(array_labels)

    plt.ylabel(y_axis)
    plt.legend()
    ax.grid(False)

    plt.tick_params(
        axis='both',
        which='both',
        width=10,
        length=5,
        direction='out'
    )

    plt.tick_params(axis='y', which='both', length=0)

    ax.set_ylabel(y_axis, fontsize=22, fontweight='bold')

    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')

    for label in ax.get_xticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')

    plt.savefig(f'{prefix}{printable_metric}.pdf', format='pdf')

def podcast_wise_results():
    with (
        open(os.path.join('/Users/naquee.rizwan/Desktop/fear-speech/podcast/toxic_conversation_chains_band_0.7_to_1'
                          '.0_sampled_at_60_seconds.json'), 'r')
    ) as data_string:
        data = json.load(data_string)

        hash_map = {}
        for chain in tqdm(data["chains"]):
            podcast_name = chain["Podcast File Name"].split("-")[0].replace("_", " ")
            if podcast_name not in hash_map:
                hash_map[podcast_name] = 0
            hash_map[podcast_name] += 1

        for item in hash_map:
            print(hash_map[item])

def speaker_wise_analysis():
    with (
            open(os.path.join('/Users/naquee.rizwan/Desktop/fear-speech/podcast/toxic_conversation_chains_band_0.7_to_1'
                              '.0_sampled_at_60_seconds.json'), 'r')
    ) as data_string:
        data = json.load(data_string)

        hash_map_array = []
        toxic_speaker = []
        for chain in tqdm(data["chains"]):
            hash_map = {"Most Speaking Speaker": chain["Most Speaking Speaker"], "speakers": {}}
            for element in chain["Before Conversations"]:
                if element["speaker"] not in hash_map["speakers"]:
                    hash_map["speakers"][element["speaker"]] = 0
                hash_map["speakers"][element["speaker"]] += round(element["end_time"] -  element["start_time"], 2)

            for element in chain["Toxic Conversation"]:
                toxic_speaker.append(hash_map["Most Speaking Speaker"] == element["speaker"])
                if element["speaker"] not in hash_map["speakers"]:
                    hash_map["speakers"][element["speaker"]] = 0
                hash_map["speakers"][element["speaker"]] += round(element["end_time"] -  element["start_time"], 2)

            for element in chain["After Conversations"]:
                if element["speaker"] not in hash_map["speakers"]:
                    hash_map["speakers"][element["speaker"]] = 0
                hash_map["speakers"][element["speaker"]] += round(element["end_time"] -  element["start_time"], 2)

            hash_map_array.append(hash_map)

        counters = [0, 0]
        for chain in toxic_speaker:
            if chain:
                counters[0] += 1
            else:
                counters[1] += 1
        # print(round((counters[0]/len(toxic_speaker))* 100, 2))

        counter_plus = 0
        counter_maxi = [0, 0]
        for chain in hash_map_array:
            if len(chain["speakers"]) > 6:
                counter_plus += 1
            maxi = 0.0
            speaker_maxi = "Unknown"
            for speaker in chain["speakers"]:
                if chain["speakers"][speaker] > maxi:
                    maxi = chain["speakers"][speaker]
                    speaker_maxi = speaker

            if speaker_maxi == chain["Most Speaking Speaker"]:
                counter_maxi[0] += 1
            else:
                counter_maxi[1] += 1

        print(round((counter_maxi[0] / (counter_maxi[0] + counter_maxi[1])) * 100, 2))
        # print(round((counter_plus/len(hash_map_array)) * 100, 2))

def episode_analysis():
    hash_map = {
        "Bannon s War Room": [],
        "Bill O Reilly s No Spin News and Analysis": [],
        "The Sean Hannity Show": [],
        "The Glenn Beck Program": [],
        "The Dan Bongino Show": [],
        "Mark Levin Podcast": [],
        "Human Events Daily with Jack Posobiec": [],
        "Conservative Review with Daniel Horowitz": [],
        "The Rubin Report": [],
        "The News Why It Matters": [],
        "The Megyn Kelly Show": [],
        "Tim Pool Daily Show": [],
        "The Ben Shapiro Show": [],
        "The New Abnormal": [],
        "Louder with Crowder": [],
        "The Charlie Kirk Show": [],
        "The Michael Savage Show": [],
        "The Jordan B Peterson Podcast": [],
        "The Michael Knowles Show": [],
        "Verdict with Ted Cruz": [],
        "Hold These Truths with Dan Crenshaw": [],
        "Bret Weinstein DarkHorse Podcast": [],
        "Pseudo Intellectual with Lauren Chen": [],
        "Fireside Chat with Dennis Prager": [],
        "Conversations With Coleman": [],
        "The One w Greg Gutfeld": [],
        "Candace Owens": [],
        "Get Off My Lawn Podcast w Gavin McInnes": [],
        "The MeidasTouch Podcast": [],
        "The Matt Walsh Show": [],
        "Rudy Giuliani s Common Sense": []
    }

    with (
            open(os.path.join('/Users/naquee.rizwan/Desktop/fear-speech/podcast/toxic_conversation_chains_band_0.7_to_1'
                              '.0_sampled_at_60_seconds.json'), 'r')
    ) as data_string:
        data = json.load(data_string)

        for chain in tqdm(data["chains"]):
            podcast_channel = chain["Podcast File Name"].split("-")[0].replace("_", " ")
            if chain["Podcast File Name"] not in hash_map[podcast_channel]:
                hash_map[podcast_channel].append(chain["Podcast File Name"])

        for podcast in hash_map:
            print(len(hash_map[podcast]))

# store_files(file_path="control_chains.json", output_file_path="control_chain_wise_map.json")
gather_statistics("Time coverage (s)", "time_coverage", input_file="chain_wise_map.json", prefix="")
# podcast_wise_results()
# speaker_wise_analysis()
# episode_analysis()
