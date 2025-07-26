import pandas
from ast import literal_eval
from ruptures.metrics import precision_recall
from ruptures.metrics import hausdorff
from ruptures.metrics import randindex
import numpy as np
import json
import os
import ast

annotator_1 = "Annotator_ND"
annotator_2 = "Annotator_SS"
annotator_3 = "Annotator_RPD"
cpd_data = pandas.read_csv("/Users/naquee.rizwan/Desktop/toxicity_begets_toxicity/cpd/dataset/cpd_annotation_conservative.csv")

cpd_data[annotator_1] = cpd_data[annotator_1].apply(literal_eval)
cpd_data[annotator_2] = cpd_data[annotator_2].apply(literal_eval)
cpd_data[annotator_3] = cpd_data[annotator_3].apply(literal_eval)

cpd_data["Inter_Annotator"] = cpd_data["Inter_Annotator"].apply(literal_eval)
cpd_data["Pelt"] = cpd_data["Pelt"].apply(literal_eval)
cpd_data["KernelCPD"] = cpd_data["KernelCPD"].apply(literal_eval)
cpd_data["BottomUp"] = cpd_data["BottomUp"].apply(literal_eval)
cpd_data["Binseg"] = cpd_data["Binseg"].apply(literal_eval)
cpd_data["KernelCPD_Detoxify_Original"] = cpd_data["KernelCPD_Detoxify_Original"].apply(literal_eval)
cpd_data["KernelCPD_Detoxify_Unbiased"] = cpd_data["KernelCPD_Detoxify_Unbiased"].apply(literal_eval)

def sanity():
    assert (
        len(cpd_data[annotator_1]) == len(cpd_data[annotator_2])
        == len(cpd_data[annotator_3]) == len(cpd_data["Inter_Annotator"])
    )
    assert (
        len(cpd_data["Pelt"]) == len(cpd_data["KernelCPD"])
        == len(cpd_data["BottomUp"]) == len(cpd_data["Binseg"]) == len(cpd_data["Inter_Annotator"])
        == len(cpd_data["KernelCPD_Detoxify_Original"]) == len(cpd_data["KernelCPD_Detoxify_Unbiased"])
    )

    unique_points = 0

    for index, item in enumerate(cpd_data["Inter_Annotator"]):
        list_sanity = []
        for change_point in range(22):
            if (
                (change_point in cpd_data[annotator_1][index] and change_point in cpd_data[annotator_2][index])
                or (change_point in cpd_data[annotator_1][index] and change_point in cpd_data[annotator_3][index])
                or (change_point in cpd_data[annotator_2][index] and change_point in cpd_data[annotator_3][index])
            ):
                list_sanity.append(change_point)

            if (change_point in cpd_data[annotator_1][index] or change_point in cpd_data[annotator_2][index] or
                    change_point in cpd_data[annotator_3][index]):
                unique_points += 1

        # Add boundary case
        if 21 not in list_sanity:
            list_sanity.append(21)
        if 21 not in item:
            item.append(21)
        assert list_sanity == item

    print("Total unique points:", unique_points)

def total_chain_distribution():
    hashmap = {

    }
    counter = 0

    for annotator in [cpd_data[annotator_1], cpd_data[annotator_2], cpd_data[annotator_3]]:
        for item in annotator:
            counter += len(item)
            if len(item) not in hashmap:
                hashmap[len(item)] = 0
            hashmap[len(item)] += 1

    keys = list(hashmap.keys())
    keys.sort()
    hashmap = {key: hashmap[key] for key in keys}

    print("Total number of annotated points:", counter)
    print("Change point size across all annotations spanning all annotators:", hashmap)

    counter = 0
    for indexer, item in enumerate(cpd_data["Inter_Annotator"]):
        counter += len(item)

    print("Total number of inter annotated points:", counter)

def calculate_metrics(dictionary_mean, dictionary_median, cost_function, margins):
    llm_data = {}
    if cost_function.split(".")[-1] == "jsonl":
        with open(cost_function, 'r') as cost_function_file:
            for line in cost_function_file:
                json_data = json.loads(line)
                for item in json_data:
                    llm_data[int(item)] = ast.literal_eval(json_data[item])

    values = {
        "precision": {},
        "recall": {},
        "hausdorff": [],
        "rand_index": []
    }
    for margin in margins:
        values["precision"][margin] = []
        values["recall"][margin] = []

    for index in range(len(cpd_data["Toxic_Chain_Index"])):
        ground_truth = cpd_data["Inter_Annotator"][index]

        if cost_function in ["Pelt", "KernelCPD", "BottomUp", "Binseg", "KernelCPD_Detoxify_Original",
                "KernelCPD_Detoxify_Unbiased"]:
            prediction = cpd_data[cost_function][index]
        else:
            assert cost_function.split(".")[-1] == "jsonl"
            prediction = []
            if int(cpd_data["Toxic_Chain_Index"][index]) in llm_data:
                prediction = llm_data[int(cpd_data["Toxic_Chain_Index"][index])]

        if len(prediction) == 0 and cost_function.split(".")[-1] == "jsonl" and "qwen" in cost_function:
            continue

        # Resolve: ruptures.metrics.sanity_check.BadPartitions: The end of the last regime is not the same for each of the partitions:
        boundary_point = 21
        if boundary_point not in ground_truth:
            assert ground_truth[-1] < boundary_point
            ground_truth.append(boundary_point)
        if boundary_point not in prediction:
            assert prediction[-1] < boundary_point
            prediction.append(boundary_point)

        for margin in margins:
            precision, recall = precision_recall(ground_truth, prediction, margin=margin)
            values["precision"][margin].append(precision)
            values["recall"][margin].append(recall)

        if len(ground_truth) > 1 and len(prediction) > 1:
            hausdorff_val = hausdorff(ground_truth, prediction)
        else:
            # boundary_point acts as infinity in our case
            hausdorff_val = boundary_point

        values["hausdorff"].append(hausdorff_val)

        rand_index_val = randindex(ground_truth, prediction)
        values["rand_index"].append(rand_index_val)

    assert (len(values["hausdorff"]) == len(values["rand_index"]) ==
            len(values["precision"][1]) == len(values["precision"][2]) == len(values["precision"][4]) ==
            len(values["recall"][1]) == len(values["recall"][2]) == len(values["recall"][4]))

    dictionary_mean["hausdorff"].append(np.mean(values["hausdorff"]))
    dictionary_median["hausdorff"].append(np.median(values["hausdorff"]))

    dictionary_mean["rand_index"].append(np.mean(values["rand_index"]))
    dictionary_median["rand_index"].append(np.median(values["rand_index"]))

    for margin in margins:
        dictionary_mean["precision"][margin].append(np.mean(values["precision"][margin]))
        dictionary_median["precision"][margin].append(np.median(values["precision"][margin]))

    for margin in margins:
        dictionary_mean["recall"][margin].append(np.mean(values["recall"][margin]))
        dictionary_median["recall"][margin].append(np.median(values["recall"][margin]))

def print_metrics(metric, mean_array, median_array):
    print(f"{metric}_mean", ": ", round(np.mean(mean_array), 2), " (", round(np.std(mean_array), 2), ")", sep="")
    print(f"{metric}_median", ": ", round(np.mean(median_array), 2), " (", round(np.std(median_array), 2), ")", sep="")

run_numbers = [1, 2, 3]

for cost in ["Pelt", "KernelCPD", "BottomUp", "Binseg", "KernelCPD_Detoxify_Original", "KernelCPD_Detoxify_Unbiased",
             "qwen-vn.jsonl", "qwen-cot.jsonl", "qwen-a-vn.jsonl", "qwen-a-cot.jsonl",
             "gpt4o-vn.jsonl", "gpt4o-cot.jsonl", "gpt4o-a-vn.jsonl", "gpt4o-a-cot.jsonl"]:
    print()
    print("Cost function:", cost)
    print()

    dictionary_map_mean = {
        "hausdorff": [],
        "rand_index": [],
        "precision": {1: [], 2: [], 4: []},
        "recall": {1: [], 2: [], 4: []}
    }

    dictionary_map_median = {
        "hausdorff": [],
        "rand_index": [],
        "precision": {1: [], 2: [], 4: []},
        "recall": {1: [], 2: [], 4: []}
    }

    if cost in ["Pelt", "KernelCPD", "BottomUp", "Binseg", "KernelCPD_Detoxify_Original",
                "KernelCPD_Detoxify_Unbiased"]:
        calculate_metrics(dictionary_map_mean, dictionary_map_median, cost_function=cost, margins=[1, 2, 4])
    else:
        for run_number in run_numbers:
            start_path = f"llm_output_run_{run_number}"
            calculate_metrics(dictionary_map_mean, dictionary_map_median, cost_function=os.path.join(start_path, cost), margins=[1, 2, 4])

    print_metrics("hausdorff", dictionary_map_mean["hausdorff"], dictionary_map_median["hausdorff"])
    print_metrics("rand_index", dictionary_map_mean["rand_index"], dictionary_map_median["rand_index"])

    print_metrics("precision_1", dictionary_map_mean["precision"][1], dictionary_map_median["precision"][1])
    print_metrics("precision_2", dictionary_map_mean["precision"][2], dictionary_map_median["precision"][2])
    print_metrics("precision_4", dictionary_map_mean["precision"][4], dictionary_map_median["precision"][4])

    print_metrics("recall_1", dictionary_map_mean["recall"][1], dictionary_map_median["recall"][1])
    print_metrics("recall_2", dictionary_map_mean["recall"][2], dictionary_map_median["recall"][2])
    print_metrics("recall_4", dictionary_map_mean["recall"][4], dictionary_map_median["recall"][4])

for cost in ["gpt4o-a-cot-detoxify-original.jsonl", "gpt4o-a-cot-detoxify-unbiased.jsonl",
             "gpt4o-a-cot-hate-xplain.jsonl", "gpt4o-a-vn-detoxify-original.jsonl",
             "gpt4o-a-vn-detoxify-unbiased.jsonl", "gpt4o-a-vn-hate-xplain.jsonl",
             "gpt4o-cot-detoxify-original.jsonl", "gpt4o-cot-detoxify-unbiased.jsonl", "gpt4o-cot-hate-xplain.jsonl",
             "gpt4o-vn-detoxify-original.jsonl", "gpt4o-vn-detoxify-unbiased.jsonl", "gpt4o-vn-hate-xplain.jsonl"]:
    print()
    print(cost.split(".")[0])
    print()

    dictionary_map_mean = {
        "hausdorff": [],
        "rand_index": [],
        "precision": {1: [], 2: [], 4: []},
        "recall": {1: [], 2: [], 4: []}
    }

    dictionary_map_median = {
        "hausdorff": [],
        "rand_index": [],
        "precision": {1: [], 2: [], 4: []},
        "recall": {1: [], 2: [], 4: []}
    }

    start_path = f"llm_output_run_ablation"
    calculate_metrics(dictionary_map_mean, dictionary_map_median, cost_function=os.path.join(start_path, cost),
                      margins=[1, 2, 4])

    print_metrics("hausdorff", dictionary_map_mean["hausdorff"], dictionary_map_median["hausdorff"])
    print_metrics("rand_index", dictionary_map_mean["rand_index"], dictionary_map_median["rand_index"])

    print_metrics("precision_1", dictionary_map_mean["precision"][1], dictionary_map_median["precision"][1])
    print_metrics("precision_2", dictionary_map_mean["precision"][2], dictionary_map_median["precision"][2])
    print_metrics("precision_4", dictionary_map_mean["precision"][4], dictionary_map_median["precision"][4])

    print_metrics("recall_1", dictionary_map_mean["recall"][1], dictionary_map_median["recall"][1])
    print_metrics("recall_2", dictionary_map_mean["recall"][2], dictionary_map_median["recall"][2])
    print_metrics("recall_4", dictionary_map_mean["recall"][4], dictionary_map_median["recall"][4])

# total_chain_distribution()
# sanity()
