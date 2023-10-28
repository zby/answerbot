import os
import sys
import csv
import json
import subprocess
from datetime import datetime
from react import get_answer
import openai
import traceback
import itertools
import pprint

ITERATIONS = 1

# this scripts generates all possible combinations of the values of the settings
# and runs the ITERATIONS experiments for each combination
# This might be a lot of experiments, so be careful!

# Check for command-line arguments
if len(sys.argv) > 1:
    filename = sys.argv[1]  # The name of the JSON file containing the questions

filename = 'data/hotpot_dev_pretty.json'

if 'filename' in locals():  # If the user specified a filename, load the questions from the file
    MAX_QUESTIONS = 10
    START_INDEX = 0
    # Load the questions from the file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Convert the JSON data to the format needed for the settings dictionary
    # And take only a subset of questions based on START_INDEX and MAX_QUESTIONS
    questions_list = [{"text": item["question"], "answers": [item["answer"]], "type": item["type"]}
                      for item in data[START_INDEX:START_INDEX+MAX_QUESTIONS]]
else:
    questions_list = [
        {
            "text": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
            "answers": ["3,677", "3677", "3,677 people", "3677 people", "4,000 capacity (3,677 seated)"],
            "type": "number"
        },
    ]

settings = {
    "questions": questions_list,
    "chunk_sizes": [
#        150,
#        200,
        300
    ],
    "functional_styles": [
        True,
        False
    ],
    "example_chunk_sizes": [
#        200,
        300
    ],
    "max_llm_calls": [5],
    "models": ["gpt-3.5-turbo"]
}

combinations = list(itertools.product(settings["chunk_sizes"], settings["functional_styles"],
                                      range(len(settings["questions"])),
                                      settings["example_chunk_sizes"], settings["max_llm_calls"], settings["models"]))

config_filename = 'config.json'

# Load the configuration from the config file
with open(config_filename, 'r') as config_file:
    json_config = json.load(config_file)
    openai.api_key = json_config["api_key"]

# Construct the directory name based on the current date and time
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = os.path.join("experiments", current_time)

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save constants to params.py
params_file_path = os.path.join(output_dir, "params.py")
with open(params_file_path, 'w') as params_file:
    for key, value in settings.items():
        params_file.write(f"{key} = {value}\n")

# Save git version and diff to version.txt
version_file_path = os.path.join(output_dir, "version.txt")
with open(version_file_path, 'w') as version_file:
    commit_hash = subprocess.getoutput("git rev-parse HEAD")
    version_file.write(f"Commit Hash: {commit_hash}\n\n")
    diff_output = subprocess.getoutput("git diff")
    version_file.write("Differences from last commit:\n")
    version_file.write(diff_output)

prompts_file_path = os.path.join(output_dir, "prompts.txt")
file_path = os.path.join(output_dir, "results.csv")
errors_file_path = os.path.join(output_dir, "errors.txt")

with open(file_path, 'w', newline='') as csvfile, open(errors_file_path, 'w') as error_file, open(prompts_file_path, 'w') as prompts_file:
    fieldnames = ['chunk_size', 'functional_style', 'example_chunk_size', 'max_llm_calls', 'model', 'answer', 'error', 'type', 'question_index', 'correct']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cs, fs, question_index, ecs, mi, model in combinations:
        config = {
            "chunk_size": cs,
            "functional": fs,
            "example_chunk_size": ecs,
            "max_llm_calls": mi,
            "model": model
        }
        for _ in range(ITERATIONS):
            question_data = settings["questions"][question_index]
            question_text = question_data["text"]
            question_type = question_data["type"]
            log_preamble = ('=' * 80) + f"\nQuestion: {question_text}\nConfig: {config}\n"
            try:
                answer, prompt = get_answer(question_text, config)
                correct = 1 if answer in question_data["answers"] else 0
                writer.writerow({
                    'chunk_size': cs,
                    'functional_style': fs,
                    'answer': answer,
                    'error': "",
                    'example_chunk_size': ecs,
                    'max_llm_calls': mi,
                    'model': model,
                    'type': question_type,
                    'question_index': question_index,
                    'correct': correct,
                })
                if fs:
                    prompt_text = pprint.pformat(prompt.openai_messages())
                else:
                    prompt_text = prompt.plain()
                prompts_file.write(f"{log_preamble}\nPrompt:\n{prompt_text}\n\n")
            except Exception as e:
                error_trace = traceback.format_exc()
                error_file.write(f"{log_preamble}\n{error_trace}\n\n")
                writer.writerow({
                    'chunk_size': cs,
                    'functional_style': fs,
                    'answer': "",
                    'error': "1",
                    'example_chunk_size': ecs,
                    'max_llm_calls': mi,
                    'model': model,
                    'type': question_type,
                    'question_index': question_index,
                    'correct': 0,
                })

print(f"Results saved to {file_path}")
print(f"Constants saved to {params_file_path}")
print(f"Prompts saved to {prompts_file_path}")
if os.path.getsize(errors_file_path) == 0:  # If the error file is empty, remove it.
    os.remove(errors_file_path)
else:
    print(f"Errors saved to {errors_file_path}")

print("Results:\n")
with open(file_path, 'r') as file:
    content = file.read()
    print(content)
