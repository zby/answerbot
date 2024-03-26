import csv
import json

experiment_dir =  "experiments/20240326181145"
experiment_dir = "experiments/old/20240324112840/"

def match_multi_answer(answer, multi_answer):
    total_elements = 0
    matches = []

    for multi_answer_element in multi_answer:
        correct = 0
        if isinstance(multi_answer_element, list):
            for correct_answer in multi_answer_element:
                if any(correct_answer in a for a in answer):
                    correct = 1
                    continue
        else:
            if any(multi_answer_element in a for a in answer):
                correct = 1
        matches.append(correct)

    total_elements = len(multi_answer)
    if total_elements == 0:
        return 0

    match_ratio = sum(matches) / total_elements
    return match_ratio

# Read the questions from the JSON file
with open(f'{experiment_dir}/params.json', 'r') as f:
    questions = json.load(f)['question']

# Read the results from the CSV file
results = []
with open(f'{experiment_dir}/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)


# Evaluate the results
evaluated_results = []
for result in results:
    question_id = int(result['question_index'])
    question = next((q for q in questions if q['id'] == question_id), None)
    correct = 0
    if question and result['answer'] is not None and result['answer'] != '':
        #print(f"result: {result['answer']}")
        answer = eval(result['answer'])  # Convert the string to a list
        if 'multi_answer' in question:
            multi_answer = question['multi_answer']
            correct = match_multi_answer(answer, multi_answer)
        else:
            for correct_answer in question.get('answer', []):
                if any(correct_answer == a for a in answer):
                    correct = 1
        result['correct'] = correct
    evaluated_results.append(result)

# Write the evaluated results to a new CSV file
fieldnames = evaluated_results[0].keys()
with open(f'{experiment_dir}/results_evaluated.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(evaluated_results)
