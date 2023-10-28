import os
import csv
import json
import subprocess
from datetime import datetime
from react import get_answer
import openai
import traceback
import itertools

ITERATIONS = 1

# this scripts generates all possible combinations of the values of the settings
# and runs the ITERATIONS experiments for each combination
# This might be a lot of experiments, so be careful!

settings = {
    #"questions": ["Who is older, Annie Morton or Terry Richardson?"],
    "questions": [
#        ("Were Scott Derrickson and Ed Wood of the same nationality?", ["yes", "American"]),
#        ("Who is older, Annie Morton or Terry Richardson?", ["Terry Richardson"]),
        ("The arena where the Lewiston Maineiacs played their home games can seat how many people?", ["3,677", "3677", "4,000 people, with 3,677 seated"]),
    ],
    #     "questions": ["Who is older, Annie Morton or Terry Richardson?",
#         "The arena where the Lewiston Maineiacs played their home games can seat how many people?"],
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

# Collect results and write to CSV
with open(file_path, 'w', newline='') as csvfile, open(errors_file_path, 'w') as error_file, open(prompts_file_path,
                                                                                                  'w') as prompts_file:
    fieldnames = ['chunk_size', 'functional_style', 'example_chunk_size', 'max_llm_calls', 'model', 'answer', 'error', 'question_index', 'correct']
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
            question = settings["questions"][question_index][0]
            log_preamble = ('=' * 80) + f"\nQuestion: {question}\nConfig: {config}\n"
            try:
                answer, prompt = get_answer(question, config)
                if answer in settings["questions"][question_index][1]:
                    correct = 1
                else:
                    correct = 0
                writer.writerow({
                    'chunk_size': cs,
                    'functional_style': fs,
                    'answer': answer,
                    'error': "",
                    'example_chunk_size': ecs,
                    'max_llm_calls': mi,
                    'model': model,
                    'question_index': question_index,
                    'correct': correct
                })
                prompts_file.write(f"{log_preamble}\nPrompt:\n{prompt.plain()}\n\n")
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
                    'question_index': question_index
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
