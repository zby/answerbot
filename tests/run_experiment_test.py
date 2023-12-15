import pytest
from unittest.mock import patch, Mock
import os
import shutil

from scripts.run_experiment import perform_experiments
from answerbot.prompt_builder import FunctionalPrompt

@pytest.fixture(scope="function")
def setup_and_teardown():
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Recursively remove directory
    os.makedirs(output_dir)
    yield output_dir
#    shutil.rmtree(output_dir)  # Cleanup after test

@patch('scripts.run_experiment.get_answer')  # This mocks the get_answer function
def test_perform_experiments(mock_get_answer, setup_and_teardown):
    output_dir = setup_and_teardown

    # Create a mock prompt and set its to_text return value
    mock_prompt = FunctionalPrompt([])
    mock_reactor = Mock()
    mock_reactor.prompt = mock_prompt
    mock_reactor.answer = "Mocked Answer"
    mock_reactor.finished = True
    mock_reactor.steps = 0
    mock_get_answer.return_value = mock_reactor

    # Example config for testing
    config = {
        "question": [{"text": "Test Question", "answers": ["Test Answer"], "type": "text"}],
        "chunk_size": [299],
        "prompt": ['FRP'],
        "reflection_prompt": ['A', ],
        "last_reflection": ['A', ],
        "max_llm_calls": [4],
        "model": ["gpt-3.5-turbo"]
    }

    # Perform experiments with the mocked reactor
    perform_experiments(config, output_dir)

    # Validate that files are created
    assert os.path.exists(os.path.join(output_dir, "results.csv"))
    assert os.path.exists(os.path.join(output_dir, "prompts.txt"))

    prompt_file = os.path.join(output_dir, "prompts", '0.txt')
    assert os.path.exists(prompt_file)
    with open(prompt_file, 'r') as file:
        file_content = file.read()
    prompt = eval(file_content)
    assert isinstance(prompt, FunctionalPrompt)

    # Validate the content of prompts.txt
    expected_prompt_fragments = [
        "Question: Test Question\n",
        "Config: {'chunk_size': 299, 'prompt': 'FRP', 'max_llm_calls': 4,",
        "'model': 'gpt-3.5-turbo', 'reflection_prompt': 'A', 'last_reflection': 'A'}\n",
    ]
    with open(os.path.join(output_dir, "prompts.txt"), 'r') as file:
        prompt_content = file.read()
    print(prompt_content)
    for i in expected_prompt_fragments:
        assert i in prompt_content
