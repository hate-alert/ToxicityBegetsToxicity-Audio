import os
import json
import shutil

from tqdm import tqdm

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


class Analysis:
    def __init__(self, json_file_path, threshold, attribute, storage_file, upper_threshold=1.0):
        self.json_file_path = json_file_path

        self.threshold = threshold
        self.upper_threshold = upper_threshold
        assert self.threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert self.upper_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.attribute = attribute

        self.toxic_chain_index = 1
        self.counter_toxic_chains = 0

        self.decay_avg = {}

        self.storage = storage_file
        fp = open(self.storage, 'w')
        json.dump({}, fp, indent=5)
        fp.close()

    def get_comments_statistics(self):
        toxicity_comments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        insult_comments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        profanity_comments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        threat_comments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        with open(self.json_file_path, "r") as analysis_file:
            file_contents = json.load(analysis_file)
            for files in file_contents["files"]:
                for index, _ in enumerate(files["TOXICITY"]):
                    toxicity_comments[index] += files["TOXICITY"][index]

                for index, _ in enumerate(files["INSULT"]):
                    insult_comments[index] += files["INSULT"][index]

                for index, _ in enumerate(files["PROFANITY"]):
                    profanity_comments[index] += files["PROFANITY"][index]

                for index, _ in enumerate(files["THREAT"]):
                    threat_comments[index] += files["THREAT"][index]

        print("TOXICITY:", toxicity_comments)
        print("INSULT:", insult_comments)
        print("PROFANITY:", profanity_comments)
        print("THREAT:", threat_comments)

    def sort_files_decreasing_order_in_band(self, max_number_of_files_per_podcast):
        band_scored_comments = {}
        with open(self.json_file_path, "r") as analysis_file:
            file_contents = json.load(analysis_file)
            for files in file_contents["files"]:
                # Negative for the ease of decreasing order sorting
                comment_count = -files[self.attribute][int(self.threshold * 10)]
                if self.upper_threshold != 1.0:
                    comment_count += files[self.attribute][int(self.upper_threshold * 10)]
                if comment_count not in band_scored_comments:
                    band_scored_comments[comment_count] = []
                band_scored_comments[comment_count].append(files["file_path"])

        keys = list(band_scored_comments.keys())
        keys.sort()
        band_scored_comments = {i: band_scored_comments[i] for i in keys}

        filtered_files = []
        for counter in band_scored_comments:
            if counter != 0:
                print(-counter, "Toxic comments in", len(band_scored_comments[counter]), "file/files")
                filtered_files.extend(band_scored_comments[counter])

        print("Number of files with zero toxicity with passed band: ", len(band_scored_comments[0]))
        print("Number of files with at least one toxicity with passed band: ", len(filtered_files))

        podcast_map = {

        }

        for top_file in filtered_files:
            podcast_folder = top_file.split('/')[-1].split('-')[0]
            if podcast_folder not in podcast_map:
                podcast_map[podcast_folder] = []
            podcast_map[podcast_folder].append(
                {
                    "original_folder": top_file,
                    "most_toxic_folder": os.path.join("most_toxic_files", top_file.split('/')[-1])
                }
            )

        top_files_storage = open(r"top_files_storage.txt", "w")

        for podcast in podcast_map.keys():
            counter = 0
            # print(podcast, len(podcast_map[podcast]))
            while counter < len(podcast_map[podcast]) and counter < max_number_of_files_per_podcast:
                top_files_storage.write(podcast_map[podcast][counter]["most_toxic_folder"])
                top_files_storage.write("\n")

                shutil.copy(
                    podcast_map[podcast][counter]["original_folder"],
                    podcast_map[podcast][counter]["most_toxic_folder"]
                )

                counter += 1

        top_files_storage.close()

        print(len(podcast_map))

    @staticmethod
    def get_speakers_with_time(data_file):
        speaker_with_time = {

        }
        with open(data_file, "r") as file:
            data = json.load(file)
        for conversations in data:
            for meta_file in data[conversations]:
                for conversation in meta_file['conversation']:
                    speaker = conversation['speaker']
                    start_time = conversation['start_time']
                    end_time = conversation['end_time']
                    if speaker not in speaker_with_time:
                        speaker_with_time[speaker] = 0
                    time = end_time - start_time
                    speaker_with_time[speaker] -= time

        tuple_speaker_time = []
        for speaker in speaker_with_time:
            tuple_speaker_time.append((speaker_with_time[speaker], speaker))

        tuple_speaker_time.sort()
        return tuple_speaker_time

    @staticmethod
    def divide_text(text, number_of_parts):
        words = text.split()
        if number_of_parts < 1:
            number_of_parts = 1
        k, m = divmod(len(words), number_of_parts)
        return [" ".join(words[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(number_of_parts)]

    def generate_toxic_chains(self, toxic_chains, data_file, unique_speakers, time_step):
        with open(data_file, "r") as file:
            data = json.load(file)

        for conversations in data:
            for meta_file in data[conversations]:
                storage = {
                    "file": meta_file["file"],
                    "conversation": [

                    ]
                }
                for index, conversation in enumerate(meta_file['conversation']):
                    if ((conversation["end_time"] - conversation["start_time"]) < time_step
                            or len(conversation[self.attribute]) == 0):
                        storage["conversation"].append(
                            {
                                "speaker": conversation["speaker"],
                                "start_time": conversation["start_time"],
                                "end_time": conversation["end_time"],
                                "text": conversation["text"],
                                self.attribute: conversation[self.attribute]
                            }
                        )
                    else:
                        duration = conversation["end_time"] - conversation["start_time"]
                        duration_step = int(duration / 17)
                        divided_texts = Analysis.divide_text(conversation["text"], duration_step)

                        assert (len(conversation["TOXICITY"]) == len(conversation["INSULT"])
                                == len(conversation["PROFANITY"]) == len(conversation["THREAT"]))

                        parts = round(time_step / 17)
                        # print(len(conversation[self.attribute]), duration_step, parts)

                        if len(divided_texts) != len(conversation[self.attribute]):
                            print(len(divided_texts), len(conversation[self.attribute]))

                        start_time = conversation["start_time"]
                        counter = 0
                        while counter <= len(conversation[self.attribute]) - 1:
                            temp_dict = {
                                "speaker": conversation["speaker"],
                                "start_time": start_time,
                                "end_time": min(conversation["end_time"], start_time + parts * 17)
                            }
                            start_time = min(conversation["end_time"], start_time + parts * 17)

                            updated_counter = min(counter + parts, len(conversation[self.attribute]))

                            # print(len(divided_texts), counter, updated_counter)
                            temp_dict["text"] = ' '.join(divided_texts[counter: updated_counter])
                            temp_dict[self.attribute] = conversation[self.attribute][counter: updated_counter]
                            counter += parts

                            storage["conversation"].append(temp_dict)

                for index, conversation in enumerate(storage["conversation"]):
                    if (conversation['speaker'] in unique_speakers and self.threshold <= max(
                            conversation[self.attribute], default=0) < self.upper_threshold):
                        toxic_chain = {
                            "Podcast File Name": storage["file"].split('/')[-1],
                            "Most Speaking Speaker": unique_speakers[0],
                            "Toxic Chain Index": self.toxic_chain_index,
                            f"Before Conversations": [],
                            f"Toxic Conversation": [storage['conversation'][index]],
                            f"After Conversations": []
                        }

                        for i in range(10, 0, -1):
                            ind = index - i
                            if ind < 0:
                                continue
                            if storage['conversation'][ind]['speaker'] in unique_speakers:
                                toxic_chain[f"Before Conversations"].append(storage['conversation'][ind])

                        for i in range(1, 11, 1):
                            ind = index + i
                            if ind >= len(storage['conversation']):
                                continue

                            if storage['conversation'][ind]['speaker'] in unique_speakers:
                                toxic_chain[f"After Conversations"].append(storage['conversation'][ind])

                        if (len(toxic_chain[f"Before Conversations"]) != 10 or len(toxic_chain[f"After Conversations"])
                                != 10):
                            self.counter_toxic_chains += 1
                            # print(len(toxic_chain[f"Before Conversations"]), len(toxic_chain[f"After Conversations"]))

                        toxic_chains["chains"].append(toxic_chain)
                        self.toxic_chain_index += 1

    def podcast_based_decay(self, podcast_channel, toxicity, speakers, unique_speakers, time_step):

        with open(self.storage, 'r') as fp:
            podcast_channels = json.load(fp)

        if podcast_channel not in podcast_channels:
            podcast_channels[podcast_channel] = {}

        if str(time_step) not in podcast_channels[podcast_channel]:
            podcast_channels[podcast_channel][str(time_step)] = {
                "Index: -10": [],
                "Index: -9": [],
                "Index: -8": [],
                "Index: -7": [],
                "Index: -6": [],
                "Index: -5": [],
                "Index: -4": [],
                "Index: -3": [],
                "Index: -2": [],
                "Index: -1": [],
                "Index: 0": [],
                "Index: 1": [],
                "Index: 2": [],
                "Index: 3": [],
                "Index: 4": [],
                "Index: 5": [],
                "Index: 6": [],
                "Index: 7": [],
                "Index: 8": [],
                "Index: 9": [],
                "Index: 10": []
            }

            for index, data_sample in enumerate(toxicity):
                if self.threshold <= data_sample < self.upper_threshold and speakers[index] in unique_speakers:
                    for ind in range(-10, 11):
                        sampled_index = index + ind
                        if ind == 0:
                            assert toxicity[sampled_index] == toxicity[index] and speakers[sampled_index] == speakers[
                                index]
                        if sampled_index < 0 or sampled_index >= len(toxicity):
                            continue
                        if speakers[sampled_index] in unique_speakers:
                            podcast_channels[podcast_channel][str(time_step)]["Index: " + str(ind)].append(
                                toxicity[sampled_index])

        with open(self.storage, 'w') as fp:
            json.dump(podcast_channels, fp, indent=5)

    def avg_decay(self, average_decay_file):
        with open(self.storage, 'r') as fp:
            podcasts = json.load(fp)

        for podcast in podcasts.keys():
            if podcast not in self.decay_avg:
                self.decay_avg[podcast] = {}

            for time_step in podcasts[podcast].keys():
                self.decay_avg[podcast][time_step] = {}
                for index in podcasts[podcast][time_step].keys():
                    if len(podcasts[podcast][time_step][index]) == 0:
                        self.decay_avg[podcast][time_step][index] = 0
                        continue
                    self.decay_avg[podcast][time_step][index] = sum(podcasts[podcast][time_step][index]) / len(
                        podcasts[podcast][time_step][index])

        with open(average_decay_file, 'w') as fp:
            json.dump(self.decay_avg, fp, indent=5)

    def decay_graph(self, graph_file):
        pdf_name = graph_file
        podcast_graphs = PdfPages(pdf_name)
        for podcast in self.decay_avg.keys():
            fig = plt.figure(figsize=(12, 6))
            for time_step in self.decay_avg[podcast].keys():
                x = []
                y = []
                for index in self.decay_avg[podcast][time_step]:
                    x.append(index[7:])
                    y.append(self.decay_avg[podcast][time_step][index])
                plt.plot(x, y, label=f'time step: {time_step}')

            plt.legend()
            plt.title('Podcast: ' + podcast)
            plt.xlabel('Index')
            plt.ylabel('Toxicity Value (average)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.83)

            podcast_graphs.savefig(fig)
        podcast_graphs.close()

    def generate_toxic_chains_in_band(self, top_files_storage, max_num_of_speakers, time_step):
        with open(top_files_storage, "r") as top_files:
            file_names = top_files.read().split("\n")

        file_names.remove("")

        # Plot the time series graph with different colors for each region
        # https://sashamaps.net/docs/resources/20-colors/

        # colors = [
        #     "Red", "Green", "Yellow", "Blue", "Orange", "Purple", "Cyan", "Magenta", "Lime", "Pink",
        #     "Teal", "Lavender", "Brown", "Beige", "Maroon", "Olive", "Navy", "Grey", "Black"
        # ]

        # pdf_name = str(time_step) + "_seconds_" + self.attribute + ".pdf"
        # podcast_graphs = PdfPages(pdf_name)

        toxic_chains = {
            "chains": []
        }

        for file_json in tqdm(file_names):
            with open(file_json, 'r', encoding="utf-8") as json_file:
                data_dict = json.load(json_file)

            segments = data_dict.get("conversations", [])

            start_time = []
            end_time = []
            toxicity = []
            speakers = []

            # Process each segment in the conversations
            for segment in segments:
                conversation = segment.get("conversation", [])
                for index, seg in enumerate(conversation):
                    if self.attribute in seg and len(seg[self.attribute]) > 0:

                        duration_step = (seg["end_time"] - seg["start_time"]) / len(seg[self.attribute])
                        # print(duration_step)

                        parts = int(round(time_step / 17.0))

                        # print(parts, len(seg[self.attribute]), seg["start_time"], seg["end_time"])
                        # print("-------")
                        counter = 0
                        start_time_custom = seg["start_time"]

                        while counter <= len(seg[self.attribute]) - 1:
                            if counter + parts - 1 <= len(seg[self.attribute]) - 1:
                                end_time_custom = start_time_custom + parts * duration_step
                                # print(counter, counter + parts - 1, start_time_custom, end_time_custom)
                                # print("######")

                                mid_time = (start_time_custom + end_time_custom) / 2.0
                                start_time_item = round(mid_time - 1)
                                end_time_item = round(mid_time + 1)

                                start_time.append(start_time_item)
                                end_time.append(end_time_item)

                                toxicity.append(max(seg[self.attribute][counter:counter + parts]))
                                speakers.append(seg["speaker"])
                                start_time_custom = end_time_custom
                            else:
                                end_time_custom = seg["end_time"]
                                # if counter > 0:
                                #     print(counter, len(seg[self.attribute]) - 1, start_time_custom, end_time_custom)
                                #     print("######")

                                mid_time = (start_time_custom + end_time_custom) / 2.0
                                start_time_item = round(mid_time - 1)
                                end_time_item = round(mid_time + 1)

                                start_time.append(start_time_item)
                                end_time.append(end_time_item)

                                toxicity.append(max(seg[self.attribute][counter:len(seg[self.attribute])]))
                                speakers.append(seg["speaker"])
                                start_time_custom = end_time_custom
                            counter += parts

                        assert len(start_time) == len(end_time) == len(toxicity) == len(speakers)
                    elif self.attribute in seg and len(seg[self.attribute]) == 0:
                        pass
                    else:
                        Warning("Something is wrong. " + self.attribute + " is not present in segment")

            speaker_list = Analysis.get_speakers_with_time(file_json)
            unique_speakers = []
            for (_, speaker) in speaker_list:
                unique_speakers.append(speaker)
            # print(speaker_list, unique_speakers)
            # print("$$$$$")

            self.generate_toxic_chains(
                toxic_chains,
                file_json,
                unique_speakers[0:max_num_of_speakers],
                time_step
            )

            self.podcast_based_decay(
                data_dict['conversations'][0]['file'].split('/')[-1].split('-')[0],
                toxicity,
                speakers,
                unique_speakers[0:max_num_of_speakers],
                time_step
            )

            # maximum_time = max(end_time)
            # time_stamps = []
            # indexer = 0
            # while indexer <= maximum_time:
            #     time_stamps.append(indexer)
            #     indexer += 1
            #
            # matrix_for_plotting = np.empty(shape=(len(time_stamps), len(unique_speakers)))
            # matrix_for_plotting.fill(0.0)
            #
            # Create a DataFrame
            # matrix_for_plotting_data_frame = pd.DataFrame(matrix_for_plotting, index=time_stamps,
            #                                               columns=unique_speakers)
            #
            # for index, _ in enumerate(speakers):
            #     start = start_time[index]
            #     end = end_time[index]
            #
            #     while start <= end:
            #         matrix_for_plotting_data_frame.loc[start, speakers[index]] = toxicity[index]
            #         start += 1
            #
            # fig = plt.figure(figsize=(12, 6))
            #
            # max_iterations = min(max_num_of_speakers, len(unique_speakers))
            # for index, _ in enumerate(unique_speakers[:max_iterations]):
            #     plt.fill_between(
            #         matrix_for_plotting_data_frame.index[0:maximum_time],
            #         matrix_for_plotting_data_frame.iloc[0:maximum_time, index],
            #         color=colors[index],
            #         alpha=0.6,
            #         label=matrix_for_plotting_data_frame.columns[index]
            #     )
            #
            # plt.axhline(y=0.2, color='gray', linestyle=':', linewidth=1)
            # plt.axhline(y=0.4, color='gray', linestyle=':', linewidth=1)
            # plt.axhline(y=0.6, color='gray', linestyle=':', linewidth=1)
            # plt.axhline(y=0.8, color='gray', linestyle=':', linewidth=1)
            #
            # plt.legend()
            # plt.title('File Name: ' + file_json.split('/')[-1].replace(".json", ""))
            # plt.xlabel('Time (Seconds)')
            # plt.ylabel('Toxicity Value')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.subplots_adjust(right=0.83)
            #
            # podcast_graphs.savefig(fig)

        with open(os.path.join("../podcast", f'toxic_conversation_chains_band_{self.threshold}_to_{self.upper_threshold}_sampled_at'
                                             f'_{time_step}_seconds_liberal_extended.json'),
                  'w') as fp:
            json.dump(toxic_chains, fp, indent=5)
        fp.close()

        # print(len(toxic_chains["chains"]))
        # print(self.counter_toxic_chains)
        # print(self.toxic_chain_index)
        #
        # podcast_graphs.close()


INFINITY = 100000000

analysis = Analysis(
    json_file_path=os.path.join("../result", "perspective_api_analysis_17_sec_liberal_extended.json"),
    threshold=0.7,
    attribute="TOXICITY",
    storage_file='podcast_decay.json',
    upper_threshold=1.0
)

analysis.get_comments_statistics()

analysis.sort_files_decreasing_order_in_band(max_number_of_files_per_podcast=INFINITY)

# Memory might run out if run in a single loop. Alternatively time step can be changed manually and file will be updated
for sample_rate in [60]:
    analysis.generate_toxic_chains_in_band(
        top_files_storage="top_files_storage.txt",
        max_num_of_speakers=INFINITY,
        time_step=sample_rate
    )

analysis.avg_decay(
    average_decay_file='avg_decay.json'
)

analysis.decay_graph(graph_file='toxicity_decay_graph.pdf')

# print(analysis.counter_toxic_chains)
