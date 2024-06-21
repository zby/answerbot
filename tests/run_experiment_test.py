import pytest
from unittest.mock import patch, Mock
import os
import shutil

from scripts.run_experiment import perform_experiments

@pytest.fixture(scope="function")
def setup_and_teardown():
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Recursively remove directory
    os.makedirs(output_dir)
    yield output_dir
#    shutil.rmtree(output_dir)  # Cleanup after test

@patch('scripts.run_experiment.create_reactor')  # This mocks the get_answer function
def test_perform_experiments(mock_create_reactor, setup_and_teardown):
    output_dir = setup_and_teardown

    # Create a mock prompt and set its to_text return value
    mock_prompt = []
    mock_reactor = Mock()
    mock_reactor.conversation = mock_prompt
    mock_reactor.answer = "Mocked Answer"
    mock_reactor.soft_errors = []
    mock_reactor.steps = 0
    mock_create_reactor.return_value = mock_reactor

    question = {
        "id": 0,
        "text": "Test Question",
        "answer": ["88.8%", "approximately 88.8%"],
        "type": "1"
    }


    # Example config for testing
    config = {
        "question": [question],
        "chunk_size": [400, 800],
        "max_llm_calls": [8, 12],
        "model": ["gpt-3.5-turbo-0613"],
        "question_check": ['category_and_amb'],
    }

    # Perform experiments with the mocked reactor
    perform_experiments(config, output_dir)

    # Validate that files are created
    assert os.path.exists(os.path.join(output_dir, "results.csv"))
    assert os.path.exists(os.path.join(output_dir, "prompts.txt"))

    # Validate the content of prompts.txt
    expected_prompt_fragments = [
        "Question: Test Question\n",
        "Config: {'chunk_size': 400, 'max_llm_calls': 8,",
        "Config: {'chunk_size': 400, 'max_llm_calls': 12,",
        "Config: {'chunk_size': 800, 'max_llm_calls': 8,",
        "Config: {'chunk_size': 800, 'max_llm_calls': 12,",
    ]
    with open(os.path.join(output_dir, "prompts.txt"), 'r') as file:
        prompt_content = file.read()
    #print(prompt_content)
    for i in expected_prompt_fragments:
        assert i in prompt_content
