import unittest
from unittest.mock import patch, Mock
import os

from scripts.run_experiment import perform_experiments
import unittest
from unittest.mock import patch, Mock
import os
import shutil

# Assuming the refactored script with perform_experiments function is named "experiment_script.py"
from scripts.run_experiment import  perform_experiments


class TestPerformExperiments(unittest.TestCase):

    # This setup runs before every test
    def setUp(self):
        self.output_dir = "test_output"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Recursively remove directory
        os.makedirs(self.output_dir)

    # This cleanup runs after every test
    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Recursively remove directory

    @patch('scripts.run_experiment.get_answer')  # This mocks the get_answer function
    def test_perform_experiments(self, mock_get_answer):
        # Create a mock prompt and set its to_text return value
        mock_prompt = Mock()
        mock_prompt.to_text.return_value = "Mocked Prompt Text"

        # Set the return value of get_answer to a tuple with the mocked answer and the mock prompt
        mock_get_answer.return_value = ("Mocked Answer", mock_prompt)

        # Example config for testing
        config = {
            "question": [{"text": "Test Question", "answers": ["Test Answer"], "type": "text"}],
            "chunk_size": [300],
            "prompt": ['FRP'],
            "example_chunk_size": [300],
            "max_llm_calls": [5],
            "model": ["gpt-3.5-turbo"]
        }

        # Perform experiments with the mocked get_answer and to_text
        perform_experiments(config, self.output_dir)

        # Validate that files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "prompts.txt")))

        # Validate the content of prompts.txt
        expected_prompt_fragments = [
            "Question: Test Question\n",
            "Config: {'chunk_size': 300, 'prompt': 'FRP', 'example_chunk_size': 300, 'max_llm_calls': 5, 'model': 'gpt-3.5-turbo'}\n",
            "Prompt:\nMocked Prompt Text\n"
        ]
        with open(os.path.join(self.output_dir, "prompts.txt"), 'r') as file:
            prompt_content = file.read()
            for i in expected_prompt_fragments:
                self.assertIn(i, prompt_content)


if __name__ == "__main__":
    unittest.main()


