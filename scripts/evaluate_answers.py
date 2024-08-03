import csv
import json
import re
import string
import sys
from collections import Counter

from answerbot.chat import Chat
from answerbot.qa_prompts import PostProcess

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def make_chat(model) -> Chat:
    template_dirs = ['answerbot/templates/common/']
    chat = Chat(
        model=model,
        one_tool_per_step=True,
        templates_dirs=template_dirs,
    )
    return chat


def preprocess_answer(answer, question, model):
    print(f"Original answer: {answer}")
    # Check if the answer is just one word
    if len(answer.split()) == 1:
        print(f'One word answer - not processing: {answer}')
        return answer

    chat = make_chat(model)
    postprocess_prompt = PostProcess(answer, question)
    result = chat(postprocess_prompt)
    if result.startswith('Implicit: '):
        result = result[9:]  # Remove the 'Compressed: ' prefix
    print(f'Processed answer: {result}')
    return result


def match_multi_answer(answer, multi_answer, use_normalization=False):
    total_elements = 0
    matches = []

    for multi_answer_element in multi_answer:
        correct = 0
        if isinstance(multi_answer_element, list):
            for correct_answer in multi_answer_element:
                if use_normalization:
                    correct_answer = normalize_answer(correct_answer)
                    normalized_answers = [normalize_answer(a) for a in answer]
                    if any(correct_answer in a for a in normalized_answers):
                        correct = 1
                        break
                else:
                    if any(correct_answer in a for a in answer):
                        correct = 1
                        break
        else:
            if use_normalization:
                multi_answer_element = normalize_answer(multi_answer_element)
                normalized_answers = [normalize_answer(a) for a in answer]
                if any(multi_answer_element in a for a in normalized_answers):
                    correct = 1
            else:
                if any(multi_answer_element in a for a in answer):
                    correct = 1
        matches.append(correct)

    total_elements = len(multi_answer)
    if total_elements == 0:
        return 0

    match_ratio = sum(matches) / total_elements
    return match_ratio

def calculate_f1(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

if len(sys.argv) < 2:
    print("Usage: python evaluate_answers.py <experiment_directory>")
    sys.exit(1)

experiment_dir = sys.argv[1]

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
    question_id = result['question_id']
    question = next((q for q in questions if q['id'] == question_id), None)
    exact_match = 0
    fuzzy_match = 0
    f1 = 0
    precision = 0
    recall = 0
    correct_answer = ''
    if question and result['answer'] is not None and result['answer'] != '':
        original_answer = result['answer']
        model = result['model']
        preprocessed_answer = preprocess_answer(original_answer, question, model)
        if 'multi_answer' in question:
            multi_answer = question['multi_answer']
            exact_match = match_multi_answer([preprocessed_answer], multi_answer)
            fuzzy_match = match_multi_answer([preprocessed_answer], multi_answer, use_normalization=True)
            # For F1, precision, and recall, we'll use the first correct answer in multi_answer
            first_correct_answer = multi_answer[0] if isinstance(multi_answer[0], str) else multi_answer[0][0]
            f1, precision, recall = calculate_f1(preprocessed_answer, first_correct_answer)
            correct_answer = str(multi_answer)  # Convert multi_answer to string
        else:
            correct_answers = question.get('answer', [])
            correct_answer = ', '.join(correct_answers)  # Join multiple correct answers
            for correct_ans in correct_answers:
                if correct_ans == preprocessed_answer:
                    exact_match = 1
                if normalize_answer(correct_ans) == normalize_answer(preprocessed_answer):
                    fuzzy_match = 1
                current_f1, current_precision, current_recall = calculate_f1(preprocessed_answer, correct_ans)
                if current_f1 > f1:
                    f1, precision, recall = current_f1, current_precision, current_recall
        result['original_answer'] = original_answer
        result['preprocessed_answer'] = preprocessed_answer
        result['exact_match'] = exact_match
        result['fuzzy_match'] = fuzzy_match
        result['f1'] = f1
        result['precision'] = precision
        result['recall'] = recall
        result['correct_answer'] = correct_answer
    evaluated_results.append(result)

# Write the evaluated results to a new CSV file
fieldnames = evaluated_results[0].keys()
with open(f'{experiment_dir}/results_evaluated.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(evaluated_results)

# Write the results to standard output
print("Evaluated Results:")
print(json.dumps(evaluated_results, indent=2))