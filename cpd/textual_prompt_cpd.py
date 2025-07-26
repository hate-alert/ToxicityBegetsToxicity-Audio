import json

with open('/Users/naquee.rizwan/Desktop/fear-speech/cpd/dataset/top_toxic_chains_conservatives.json', 'r') as file:
    top_toxic_chains = json.load(file)

print(len(top_toxic_chains))

def generate_skeleton(index_counter, diarized_text, toxicity):
    return str(index_counter) + ". " + str(diarized_text) + " (" + "Toxicity: " + str(round(toxicity, 2)) + ")" + "\n"

# Important note: Concatenated toxicity is the toxicity of current and previous segments combined

# type_of_prompt = "vn" or "cot"
def get_prompt(type_of_prompt, chain):

    index_counter = 0
    common_prompt = "\ncontinuous conversation segments:\n\n"
    
    for itemize in chain["Before Conversations"]:
        concatenated_toxicity = 0.0
        if "Concatenated_Toxicity" in itemize:
            concatenated_toxicity = itemize["Concatenated_Toxicity"]
        elif "TOXICITY" in itemize and len(itemize["TOXICITY"]) > 0:
            concatenated_toxicity = max(itemize["TOXICITY"])
        common_prompt += generate_skeleton(
            index_counter,
            itemize["text"],
            concatenated_toxicity
        )
        index_counter += 1

    for itemize in chain["Toxic Conversation"]:
        concatenated_toxicity = 0.0
        if "Concatenated_Toxicity" in itemize:
            concatenated_toxicity = itemize["Concatenated_Toxicity"]
        elif "TOXICITY" in itemize and len(itemize["TOXICITY"]) > 0:
            concatenated_toxicity = max(itemize["TOXICITY"])
        common_prompt += generate_skeleton(
            index_counter,
            itemize["text"],
            concatenated_toxicity
        )
        index_counter += 1

    for itemize in chain["After Conversations"]:
        concatenated_toxicity = 0.0
        if "Concatenated_Toxicity" in itemize:
            concatenated_toxicity = itemize["Concatenated_Toxicity"]
        elif "TOXICITY" in itemize and len(itemize["TOXICITY"]) > 0:
            concatenated_toxicity = max(itemize["TOXICITY"])
        common_prompt += generate_skeleton(
            index_counter,
            itemize["text"],
            concatenated_toxicity
        )
        index_counter += 1

    prefix_full_prompt = [
        "You will be provided with continuous conversation segments of a podcast with spoken text and corresponding toxicity value on a scale of 0 to 1. "
        "Your task is to detect **important** change points where the conversation's tone, toxicity level and conversation sentiment shifts **abruptly** and **drastically**. "
        "Analyze and provide an array of **important** change points based on your analysis.\n"
        "Only output an array of **important** change points without any explanation. Note that segments with toxicity value higher than 0.7 are considered highly toxic.\n"
    ][0]
    
    cot_suffix_full_prompt = ""
    
    if type_of_prompt == "vn":
        prefix_full_prompt += [
            "Reference examples of output format: [1, 5, 10] or [1, 11, 13, 17] or [9, 16, 20].\n"
            "Keep number of change points as minimum as possible and don't output more than 7 change points. Choose these change points very wisely."
        ][0]
    
    elif type_of_prompt == "cot":
        cot_suffix_full_prompt = [
            "\nConsider the following definitions while analyzing change points:\n"
            "(a) shift in tone- **abrupt** shifts in sentiment or intensity, such as transitions from neutral or mildly toxic statements to overt hostility.\n"
            "(b) topical shift- **abrupt** emergence of new discussion themes or **abrupt** cessation of previously dominant topics.\n"
            "(c) change in toxicity- **abrupt** escalations or reductions in the degree of harmful language or expressions.\n"
            "Now think step by step, analyze the provided continuous conversation segments and output only "
            "**important** change points where the conversation's tone, toxicity level and conversation sentiment shifts **abruptly** and **drastically**.\n"
            "Reference examples of output format: [1, 5, 10] or [1, 11, 13, 17] or [9, 16, 20].\n"
            "Keep number of change points as minimum as possible and don't output more than 7 change points. Choose these change points very wisely."
        ][0]
    
    else:
        raise("Invalid input")

    return prefix_full_prompt + common_prompt + cot_suffix_full_prompt

from openai import AzureOpenAI


model = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

deployment_name=''

output_list = []

for indexer, item in enumerate(top_toxic_chains):
    
    prompt = get_prompt("vn", item)
    # print(prompt)
    
    # Create system message to define the task context
    message = [
        {"role": "system", "content": "You are an expert in analyzing toxic speech in conversations. Your task is to detect **important** change points where the conversation's tone, toxicity level and conversation sentiment shifts **abruptly** and **drastically**."},
        {"role": "user", "type": "text", "content": prompt}
    ]
    
    response = model.chat.completions.create(
        model=deployment_name,
        messages=message,
        max_tokens=50,
        temperature=0.001
    )
    
    generated_text = response.choices[0].message.content

    item_dict = {item["Toxic Chain Index"]: generated_text}
    output_list.append(item_dict)
    
    print(item_dict)

with open('gpt4o-vn.jsonl', 'w') as f:
    for item_cpd in output_list:
        json_cpd = json.dumps(item_cpd)
        f.write(json_cpd + '\n')
