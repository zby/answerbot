import json
from scripts.qa_delegated import main_processor
from scripts.answer import app # Import the other processor

INPUT_FILE = 'data/beerqa_test_questions_first_10.json'
OUTPUT_FILE = 'logs/beerqa_test_questions_first_10.json'
MAX_QUESTIONS = None  # Limit the number of questions to process

def process_questions(processor):
    # Read input file
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    answers = {}

    # Process each question, up to the limit if MAX_QUESTIONS is set
    questions_to_process = data['data'][:MAX_QUESTIONS] if MAX_QUESTIONS is not None else data['data']
    for item in questions_to_process:
        question_id = item['id']
        question = item['question']
        print(f"\n\nProcessing question: {question}")
        
        answer = processor.process(question)  # Use the passed processor
        answers[question_id] = answer

    # Write answers to output file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"answer": answers}, f, indent=2)

    print(f"Processed {len(answers)} questions. Answers written to {OUTPUT_FILE}")
    print(f"Processed {len(answers)} questions. Answers written to {OUTPUT_FILE}")

if __name__ == "__main__":
#    process_questions(main_processor)  # Pass main_processor as an argument
    process_questions(app)  # Process again with answer_processor