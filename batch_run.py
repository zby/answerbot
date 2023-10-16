import json
import csv
import os
import sys
from datetime import datetime
from react import get_answer  # Import the get_answer function from react.py

# Constants
MAX_QUESTIONS = 5
config_filename = 'config.json'

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

# Load the configuration from the config file
with open(config_filename, 'r') as config_file:
    config = json.load(config_file)

# Iterate through the data and add new answers, limited by MAX_QUESTIONS
answered_questions = 0
limited_data = []
for entry in data:
    if answered_questions >= MAX_QUESTIONS:
        break

    question = entry['question']
    new_answer = get_answer(config, question)
    entry['new_answer'] = new_answer
    limited_data.append(entry)
    answered_questions += 1

# Write the updated data to the CSV file
with open(csv_filename, 'w', newline='') as csv_file:
    header = limited_data[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=header)

    writer.writeheader()
    writer.writerows(limited_data)

print(f'Answers successfully generated and saved to {csv_filename}')
