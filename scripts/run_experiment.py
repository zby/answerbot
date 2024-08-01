import os
import sys
import csv
import json
import subprocess
import time

from datetime import datetime
import itertools
import traceback
import logging

from answerbot.qa_processor import QAProcessor, QAProcessorDeep

from answerbot.tools.wiki_tool import WikipediaTool

from dotenv import load_dotenv

load_dotenv()

# Constants
ITERATIONS = 1
CONFIG_KEYS = ['processor_type', 'chunk_size', 'max_llm_calls', 'model']
ADDITIONAL_KEYS = ['question_id', 'answer', 'error', 'error_type', 'warnings']


def create_reactor(config):
    wiki_processor_config = {
        'max_iterations': config['max_llm_calls'],
        'model': config['model'],
        'prompt_templates_dirs': ['answerbot/templates/common', 'answerbot/templates/wiki_researcher'],
        'toolbox': [WikipediaTool(chunk_size=config['chunk_size'])],
        'name': 'wiki_processor'
    }
    if config['processor_type'] == 'deep':
        return QAProcessorDeep(
            toolbox=[],
            max_iterations=3,
            model=config['model'],
            prompt_templates_dirs = ['answerbot/templates/common', 'answerbot/templates/main_researcher'],
            name='main_processor',
            sub_processor_config=wiki_processor_config,
            delegate_description="Delegate a question to a Wikipedia expert."
        )
    else:
        return QAProcessor(**wiki_processor_config)


def load_questions_from_file(filename, start_index, end_index):
    with open(filename, 'r') as f:
        data = json.load(f)
    result = []
    for item in data[start_index:end_index]:
        if 'guess' in item:
            continue
        result.append(item)
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
    with open(params_file_path, "w") as file:
        json.dump(settings, file, indent=4)


def eval_question(combo, results_writer, csvfile, error_file):
    question = combo[-1]
    config = dict(zip(CONFIG_KEYS, combo[:-1]))
    config['question_id'] = question['id']

    question_text = question["text"]
    log_preamble = ('=' * 80) + f"\nQuestion: {question_text}\nConfig: {config}\n"
    retry_delay = [360]
    for delay in retry_delay + [0]:  # After last failure we don't wait because we don't repeat
        try:
            reactor = create_reactor(config)
            answer = reactor.process(question_text)

            # More robust warning counting
            qa_logger = logging.getLogger('qa_processor')
            chat_logger = logging.getLogger('answerbot.chat')

            warning_count = sum(1 for handler in qa_logger.handlers for record in handler.records
                                if record.levelno == logging.WARNING)
            chat_warning_count = sum(1 for handler in chat_logger.handlers for record in handler.records
                                     if record.levelno == logging.WARNING)
            total_warning_count = warning_count + chat_warning_count

            config.update({
                'answer': answer,
                'error': 0,
                'error_type': '',
                'warnings': total_warning_count,
            })
            results_writer.writerow(config)
            csvfile.flush()
            os.fsync(csvfile.fileno())
            break
        except Exception as e:
            error_trace = traceback.format_exc()
            error_file.write(f"{log_preamble}\n{error_trace}\n\n")
            error_file.flush()
            error_type = type(e).__name__
            config.update({
                'error': 1,
                'error_type': error_type
            })
            results_writer.writerow(config)
            csvfile.flush()
            os.fsync(csvfile.fileno())
            if error_type == 'APITimeoutError':
                print(f"APITimeoutError occurred. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                break



def perform_experiments(settings, output_dir):
    file_path = os.path.join(output_dir, "results.csv")
    errors_file_path = os.path.join(output_dir, "errors.txt")

    with open(file_path, 'w', newline='') as csvfile, open(errors_file_path, 'w') as error_file:
        writer = csv.DictWriter(csvfile, fieldnames=CONFIG_KEYS + ADDITIONAL_KEYS)
        writer.writeheader()

        # Produce combinations using the defined settings keys
        combinations = list(
            itertools.product(*[settings[key] for key in CONFIG_KEYS + ['question'] ]))

        for combo in combinations:
            for _ in range(ITERATIONS):
                eval_question(combo, writer, csvfile, error_file)

    if os.path.getsize(errors_file_path) == 0:  # If the error file is empty, remove it.
        os.remove(errors_file_path)


def record_experiment(settings, output_dir=None):
    if output_dir is None:
        output_dir = generate_directory_name()
    save_constants_to_file(os.path.join(output_dir, "params.json"), settings)
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

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    #filename = 'data/hotpot_reasonable.json'
    #filename = 'filtered_questions.json'
    if filename:
        start_index = 0
        end_index = 1
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
#            {
#                "text": "Are Random House Tower and 888 7th Avenue both used for real estate?",
#                "answers": ["no"],
#                "type": "comparison"
#            }
#            {
#                "text": "Who is older, Annie Morton or Terry Richardson?",
#                "answers": ["Terry Richardson", "Terry Richardson is older"],
#                "type": "bridge"
#            }
            {
                "id": 0,
                "text": "What is the weight proportion of oxygen in water?",
                "answer": ["88.8%", "approximately 88.8%"],
                "type": "1"
            }
        ]

    settings = {
        "processor_type": [
            "deep",
            "simple"
        ],
        "question": questions_list,
        "chunk_size": [
            400
        ],
        "max_llm_calls": [5],
        "model": [
            "claude-3-haiku-20240307",
            "gpt-4o-mini"
        ],
    }

    record_experiment(settings)