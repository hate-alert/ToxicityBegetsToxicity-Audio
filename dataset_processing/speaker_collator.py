import os
import json

folder_paths = [
    "liberal_podcast_diarized/Mea_Culpa",
    "liberal_podcast_diarized/Pod_Save_America",
    "liberal_podcast_diarized/The_MediasTouch_Podcast",
    "liberal_podcast_diarized/The_Rachel_Maddow_Show",
    "liberal_podcast_diarized/Why_is_This_Happening_with_Chris_Hayes"
]

minimum = 10000
maximum = -1

counter = 0
for podcast in folder_paths:
    for filename in os.listdir(podcast):
        file_path = os.path.join(podcast, filename)
        assert os.path.isfile(file_path) == True

        counter += 1
        print(counter, end=',')

        with open(file_path, 'r') as file:
            data = json.load(file)

        num_segments = len(data['segments'])
        minimum = min(minimum, num_segments)
        maximum = max(maximum, num_segments)

        processed_files = {'conversations': []}
        processed_files['conversations'].append({'file': file_path, 'conversation': []})

        if 'speaker' not in data['segments'][0]:
            data['segments'][0]["speaker"] = "unknown"

        total_text = data['segments'][0]["text"]
        prev_speaker = data['segments'][0]["speaker"]
        initial_start_time = data['segments'][0]["start"]
        final_end_time = data['segments'][0]["end"]

        for index, segment in enumerate(data['segments']):

            if index == 0:
                continue

            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            if 'speaker' not in segment:
                segment['speaker'] = "unknown"
            speaker = segment['speaker']

            if speaker != prev_speaker:
                segment_collated = {
                    'start_time' : initial_start_time,
                    'end_time' : final_end_time,
                    'text' : total_text,
                    'word_count' : len(total_text.split(' ')) - 1,
                    'speaker' : prev_speaker
                }

                processed_files['conversations'][0]['conversation'].append(segment_collated)

                total_text = text
                prev_speaker = speaker
                initial_start_time = start_time
                final_end_time = end_time
            else:
                total_text += f' {text}'
                final_end_time = end_time

        segment_collated = {
            'start_time': initial_start_time,
            'end_time': final_end_time,
            'text': total_text,
            'word_count': len(total_text.split(' ')) - 1,
            'speaker': prev_speaker
        }

        processed_files['conversations'][0]['conversation'].append(segment_collated)

        output_file_path = file_path.replace('liberal_podcast_diarized', 'liberal_podcast_diarized_pre_processed')
        # print(output_file_path)

        os.makedirs("/".join(output_file_path.split("/")[0:-1]), exist_ok=True)
        with open(output_file_path, "w") as file:
            json.dump(processed_files, file, indent=4)
