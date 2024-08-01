import json
import os
import sys
import csv
import glob
import shutil
from scripts.qa_delegated import main_processor
from scripts.answer import app  # Import the other processor

OUTPUT_DIR = 'logs'
MAX_QUESTIONS = None  # Limit the number of questions to process

def get_next_file_number(base_path):
    pattern = f"{base_path}_*"
    existing_files = glob.glob(pattern)
    if not existing_files:
        return 1

    numbers = [int(f.split('_')[-1]) for f in existing_files if f.split('_')[-1].isdigit()]
    return max(numbers) + 1 if numbers else 1

def process_questions(processor, input_file_path):
    # Split the input file path
    input_dir, input_filename = os.path.split(input_file_path)

    # Read input file
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    answers = {}

    # Create CSV output file
    csv_base_path = os.path.join(OUTPUT_DIR, os.path.splitext(input_filename)[0])
    csv_output_path = f"{csv_base_path}.csv"

    # Check if the file already exists and rename if necessary
    if os.path.exists(csv_output_path):
        next_number = get_next_file_number(csv_base_path)
        new_csv_path = f"{csv_base_path}_{next_number}.csv"
        shutil.move(csv_output_path, new_csv_path)
        print(f"Existing file moved to: {new_csv_path}")

    with open(csv_output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Question ID', 'Question', 'Answer'])

        # Process each question, up to the limit if MAX_QUESTIONS is set
        questions_to_process = data['data'][:MAX_QUESTIONS] if MAX_QUESTIONS is not None else data['data']
        for item in questions_to_process:
            question_id = item['id']
            question = item['question']
            print(f"\n\nProcessing question: {question}")
            try:
                answer = processor.process(question)  # Use the passed processor
            except Exception as e:
                print(f"Error processing question: {question}")
                print(f"Error details: {str(e)}")
                answer = "Error: Unable to process this question"
            answers[question_id] = answer

            # Write to CSV immediately and flush
            csv_writer.writerow([question_id, question, answer])
            csvfile.flush()  # Flush after each row

    # Write answers to JSON output file
    json_output_path = os.path.join(OUTPUT_DIR, input_filename)
    with open(json_output_path, 'w') as f:
        json.dump({"answer": answers}, f, indent=2)

    print(f"Processed {len(answers)} questions.")
    print(f"Answers written to JSON file: {json_output_path}")
    print(f"Answers written to CSV file: {csv_output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python beerqa.py <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    process_questions(main_processor, input_file_path)  # Pass main_processor as an argument
#    process_questions(app, input_file_path)  # Process again with answer_processor