import json
import csv
import os
import sys
import openai
from datetime import datetime
from react import get_answer

# Constants
MAX_QUESTIONS = 5
START_INDEX = 0

config = {
    "chunk_size": 300,
    "functional": True,
    "example_chunk_size": 200,
    "max_llm_calls": 5,
    "model": "gpt-3.5-turbo"
}

config_filename = 'config.json'
# Load the configuration from the config file
with open(config_filename, 'r') as config_file:
    json_config = json.load(config_file)
    openai.api_key = json_config["api_key"]


# Check for the correct number of command line arguments
if len(sys.argv) != 2:
    print("Usage: python batch_run.py <input_filename.json>")
    sys.exit(1)

# Command line argument for the input JSON filename
input_filename = sys.argv[1]

# Create a 'logs' directory if it doesn't exist
logs_directory = 'logs'
os.makedirs(logs_directory, exist_ok=True)

# Construct the output CSV filename based on the input filename and current date and time
input_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
csv_filename = os.path.join(logs_directory, f'{input_filename_prefix}_{timestamp}.csv')

# Load the JSON data from the file
with open(input_filename, 'r') as json_file:
    data = json.load(json_file)

# Iterate through the data and add new answers, limited by MAX_QUESTIONS
answered_questions = 0
limited_data = []
for entry in data[START_INDEX:]:
    if answered_questions >= MAX_QUESTIONS:
        break

    question = entry['question']
    try:
        new_answer, prompt = get_answer(question, config)
        entry['new_answer'] = new_answer
        entry['error'] = None
    except Exception as e:
        print(f'Error: {e}')
        entry['new_answer'] = None
        entry['error'] = 1
    limited_data.append(entry)
    answered_questions += 1

if limited_data:
    # Write the updated data to the CSV file
    with open(csv_filename, 'w', newline='') as csv_file:
        header = limited_data[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=header)

        writer.writeheader()
        writer.writerows(limited_data)

    print(f'Answers successfully generated and saved to {csv_filename}')

    print('=' * 80)
    print('=' * 80)

    with open(csv_filename, 'r') as csv_file:
        print(csv_file.read())
else:
    print('No questions found!')