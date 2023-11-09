import os
import sys
import csv
import json
import subprocess
from datetime import datetime
import itertools
import traceback
from react import get_answer
import openai

# Constants
ITERATIONS = 1
CONFIG_KEYS = ['chunk_size', 'prompt', 'example_chunk_size', 'max_llm_calls', 'model']
ADDITIONAL_KEYS = ['answer', 'error', 'type', 'question_index', 'correct']

def load_config_from_file(config_filename):
    with open(config_filename, 'r') as config_file:
        json_config = json.load(config_file)
        openai.api_key = json_config["api_key"]

def load_questions_from_file(filename, start_index, end_index):
    with open(filename, 'r') as f:
        data = json.load(f)
    result = []
    for item in data[start_index:end_index]:
        if isinstance(item['answer'], list):
            answers = item['answer']
        else:
            answers = [item['answer']]
        result.append({"text": item["question"], "answers": answers, "type": item["type"]})
    return result

def generate_directory_name():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join("experiments", current_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def save_git_version_and_diff(version_file_path):
    with open(version_file_path, 'w') as version_file:
        commit_hash = subprocess.getoutput("git rev-parse HEAD")
        version_file.write(f"Commit Hash: {commit_hash}\n\n")
        diff_output = subprocess.getoutput("git diff")
        version_file.write("Differences from last commit:\n")
        version_file.write(diff_output)


def save_constants_to_file(params_file_path, settings):
    with open(params_file_path, 'w') as params_file:
        for key, value in settings.items():
            params_file.write(f"{key} = {value}\n")


def perform_experiments(settings, output_dir):
    prompts_file_path = os.path.join(output_dir, "prompts.txt")
    file_path = os.path.join(output_dir, "results.csv")
    errors_file_path = os.path.join(output_dir, "errors.txt")

    with open(file_path, 'w', newline='') as csvfile, open(errors_file_path, 'w') as error_file, open(prompts_file_path,
                                                                                                      'w') as prompts_file:
        writer = csv.DictWriter(csvfile, fieldnames=CONFIG_KEYS + ADDITIONAL_KEYS)
        writer.writeheader()

        # Produce combinations using the defined settings keys
        combinations = list(
            itertools.product(*[settings[key] for key in CONFIG_KEYS], range(len(settings["question"]))))

        for combo in combinations:
            # Use dictionary unpacking to get the values by key and build the config dictionary
            config = dict(zip(CONFIG_KEYS, combo[:-1]))
            question_index = combo[-1]

            for _ in range(ITERATIONS):
                question_data = settings["question"][question_index]
                question_text = question_data["text"]
                question_type = question_data["type"]
                log_preamble = ('=' * 80) + f"\nQuestion: {question_text}\nConfig: {config}\n"

                try:
                    answer, prompt = get_answer(question_text, config)
                    correct = 1 if answer in question_data["answers"] else 0
                    config.update({
                        'type': question_type,
                        'question_index': question_index,
                        'answer': answer,
                        'error': "",
                        'correct': correct,
                    })
                    prompts_file.write(f"{log_preamble}\nPrompt:\n{prompt.to_text()}\n\n")
                except Exception as e:
                    error_trace = traceback.format_exc()
                    error_file.write(f"{log_preamble}\n{error_trace}\n\n")
                    config.update({
                        'type': question_type,
                        'question_index': question_index,
                        'error': 1,
                        'correct': 0,
                    })
                writer.writerow(config)
    if os.path.getsize(errors_file_path) == 0:  # If the error file is empty, remove it.
        os.remove(errors_file_path)


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    filename = 'data/hotpot_dev_pretty.json'
    #filename = 'filtered_questions.json'
    if filename:
        start_index = 0
        end_index = 10
        questions_list = load_questions_from_file(filename, start_index, end_index)
    else:
        # Default Question
        questions_list = [
#            {
#                "text": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
#                "answers": ["3,677", "3677", "3,677 people", "3677 people"],
#                "type": "number"
#            },
#            {
#                "text": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
#                "answers": ["YG Entertainment"],
#                "type": "bridge"
#            },
            {
                "text": "Are Random House Tower and 888 7th Avenue both used for real estate?",
                "answers": ["no"],
                "type": "comparison"
            }
        ]

    settings = {
        "question": questions_list,
        "chunk_size": [
            300
        ],
        "prompt": [
#            'TRP',
#            'FRP',
            'NFRP',
        ],
        "example_chunk_size": [
#            300,
            200,
        ],
        "max_llm_calls": [7],
        "model": ["gpt-4"]
    }
    load_config_from_file('config.json')
    output_dir = generate_directory_name()
    save_constants_to_file(os.path.join(output_dir, "params.py"), settings)
    save_git_version_and_diff(os.path.join(output_dir, "version.txt"))
    perform_experiments(settings, output_dir)
    for file, what in [
        ('results.csv', 'Results'), ('errors.txt', 'Errors'),
        ('params.py', 'Tested parameters'), ('prompts.txt', 'Prompts'),
        ('version.txt', 'Git Version and Diff')]:
        if os.path.exists(os.path.join(output_dir, file)):
            print(f"{what} saved to {os.path.join(output_dir, file)}")
    print("Results:\n")
    with open(os.path.join(output_dir, 'results.csv'), 'r') as file:
        content = file.read()
        print(content)
