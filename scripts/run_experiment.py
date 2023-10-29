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
FIELDNAMES = ['chunk_size', 'functional_style', 'example_chunk_size', 'max_llm_calls', 'model', 'answer', 'error',
              'type', 'question_index', 'correct']


def load_config_from_file(config_filename):
    with open(config_filename, 'r') as config_file:
        json_config = json.load(config_file)
        openai.api_key = json_config["api_key"]


def load_questions_from_file(filename, start_index, end_index):
    with open(filename, 'r') as f:
        data = json.load(f)
    return [{"text": item["question"], "answers": [item["answer"]], "type": item["type"]}
            for item in data[start_index:end_index]]


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
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        combinations = list(
            itertools.product(settings["chunk_sizes"], settings["functional_styles"], range(len(settings["questions"])),
                              settings["example_chunk_sizes"], settings["max_llm_calls"], settings["models"]))

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
                    prompts_file.write(f"{log_preamble}\nPrompt:\n{prompt.to_text()}\n\n")
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

    if os.path.getsize(errors_file_path) == 0:  # If the error file is empty, remove it.
        os.remove(errors_file_path)


def display_results(file_path, params_file_path, errors_file_path=None):
    print(f"Results saved to {file_path}")
    print(f"Constants saved to {params_file_path}")
    if errors_file_path and os.path.exists(errors_file_path):
        print(f"Errors saved to {errors_file_path}")
    print("Results:\n")
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if filename:
        start_index = 0
        end_index = 0
        questions_list = load_questions_from_file(filename, start_index, end_index)
    else:
        # Default Question
        questions_list = [
            {
                "text": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
                "answers": ["3,677", "3677", "3,677 people", "3677 people", "4,000 capacity (3,677 seated)"],
                "type": "number"
            },
        ]

    settings = {
        "questions": questions_list,
        "chunk_sizes": [300],
        "functional_styles": [True, False],
        "example_chunk_sizes": [300],
        "max_llm_calls": [5],
        "models": ["gpt-3.5-turbo"]
    }
    load_config_from_file('config.json')
    output_dir = generate_directory_name()
    save_constants_to_file(os.path.join(output_dir, "params.py"), settings)
    save_git_version_and_diff(os.path.join(output_dir, "version.txt"))
    perform_experiments(settings, output_dir)
    display_results(os.path.join(output_dir, "results.csv"), os.path.join(output_dir, "params.py"),
                    os.path.join(output_dir, "errors.txt"))
