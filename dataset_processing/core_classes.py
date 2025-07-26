import json
import os

from googleapiclient import discovery


class CommentAnalyzer:
    def __init__(self, api_key):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def analyze_text(self, text):
        """Analyzes the given text using the Comment Analyzer API."""
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}, "INSULT": {}, "THREAT": {}, "PROFANITY": {}}
        }
        return self.client.comments().analyze(body=analyze_request).execute()


class JSONParser:

    def __init__(self):
        self.base_path = "/Users/Vishwajeet/Desktop/Project"

    def extract_information_from_folder(self, folder_path):
        extracted_info = []  # To collect all extracted data
        # print("\nFolder Name:", folder_path, "\n")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".json"):
                segments, f_path = JSONParser.extract_information_from_json(file_path)
                if segments is not None:
                    extracted_info.append((segments, f_path))  # Collect data for processing later
            else:
                if filename == "analysis":
                    continue
                else:
                    if os.path.isdir(file_path):
                        extracted_info.extend(self.extract_information_from_folder(file_path))
        return extracted_info  # Return collected data for further processing

    @staticmethod
    def extract_information_from_json(file_path):
        if file_path.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data_dict = json.load(file)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return None, file_path

            if "segments" in data_dict:
                segments = data_dict["segments"]
            elif "conversations" in data_dict:
                segments = data_dict["conversations"]
            return segments, file_path
