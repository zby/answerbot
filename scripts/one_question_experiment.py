import os
import csv
import json
import subprocess
from datetime import datetime
from react import get_answer
import openai
import traceback

# Constants
QUESTION = "Who is older, Annie Morton or Terry Richardson?"
QUESTION = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
CHUNK_SIZES = [100, 150, 200, 300]
FUNCTIONAL_STYLES = [True, False]
COMBINATIONS = [(cs, fs) for cs in CHUNK_SIZES for fs in FUNCTIONAL_STYLES]
ITERATIONS = 2
config_filename = 'config.json'


# Load the configuration from the config file
with open(config_filename, 'r') as config_file:
    config = json.load(config_file)
    openai.api_key = config["api_key"]


# Construct the directory name based on the current date and time
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = os.path.join("experiments", current_time)

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save constants to params.py
params_file_path = os.path.join(output_dir, "params.py")
with open(params_file_path, 'w') as params_file:
    params_file.write(f"QUESTION = '{QUESTION}'\n")
    params_file.write(f"CHUNK_SIZES = {CHUNK_SIZES}\n")
    params_file.write(f"FUNCTIONAL_STYLES = {FUNCTIONAL_STYLES}\n")
    params_file.write(f"COMBINATIONS = {COMBINATIONS}\n")

# Save git version and diff to version.txt
version_file_path = os.path.join(output_dir, "version.txt")
with open(version_file_path, 'w') as version_file:
    commit_hash = subprocess.getoutput("git rev-parse HEAD")
    version_file.write(f"Commit Hash: {commit_hash}\n\n")

    diff_output = subprocess.getoutput("git diff")
    version_file.write("Differences from last commit:\n")
    version_file.write(diff_output)

# Define the results.csv and errors.txt file paths
file_path = os.path.join(output_dir, "results.csv")
errors_file_path = os.path.join(output_dir, "errors.txt")

# Collect results and write to CSV
with open(file_path, 'w', newline='') as csvfile, open(errors_file_path, 'w') as error_file:
    fieldnames = ['chunk_size', 'functional_style', 'answer', 'error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for cs, fs in COMBINATIONS:
        for _ in range(ITERATIONS):
            error_flag = ""
            answer = ""
            try:
                answer = get_answer(QUESTION, cs, fs)
            except Exception as e:
                error_flag = "1"
                error_trace = traceback.format_exc()
                error_file.write(f"\nError for chunk_size={cs}, functional_style={fs}:\n{error_trace}\n")

            writer.writerow({'chunk_size': cs, 'functional_style': fs, 'answer': answer, 'error': error_flag})

print(f"Results saved to {file_path}")
print(f"Constants saved to {params_file_path}")
if os.path.getsize(errors_file_path) == 0:  # If the error file is empty, remove it.
    os.remove(errors_file_path)
else:
    print(f"Errors saved to {errors_file_path}")
