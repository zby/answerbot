import unittest
from unittest.mock import patch, Mock
import os
import shutil

# Assuming the refactored script with perform_experiments function is named "experiment_script.py"
from scripts.run_experiment import perform_experiments
from answerbot.prompt_builder import FunctionalPrompt

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
        mock_prompt = FunctionalPrompt([])
        mock_reactor = Mock()
        mock_reactor.prompt = mock_prompt
        mock_reactor.answer = "Mocked Answer"
        mock_reactor.finished = True
        mock_reactor.steps = 1

        # Set the return value of get_answer to a tuple with the mocked answer and the mock prompt
        mock_get_answer.return_value = mock_reactor

        # Example config for testing
        config = {
            "question": [{"text": "Test Question", "answers": ["Test Answer"], "type": "text"}],
            "chunk_size": [300],
            "prompt": ['FRP'],
            "reflection_prompt": ['A', ],
            "last_reflection": ['A', ],
            "max_llm_calls": [5],
            "model": ["gpt-3.5-turbo"]
        }

        # Perform experiments with the mocked reactor
        perform_experiments(config, self.output_dir)

        # Validate that files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "results.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "prompts.txt")))

        prompt_file = os.path.join(self.output_dir, "prompts", '0.txt')
        self.assertTrue(os.path.exists(prompt_file))
        file = open(prompt_file, 'r')
        file_content = file.read()
        file.close()
        prompt = eval(file_content)
        self.assertIsInstance(prompt, FunctionalPrompt)

        # Validate the content of prompts.txt
        expected_prompt_fragments = [
            "Question: Test Question\n",
            "Config: {'chunk_size': 300, 'prompt': 'FRP', 'max_llm_calls': 5, 'model': 'gpt-3.5-turbo', 'reflection_prompt': 'A', 'last_reflection': 'A'}\n",
        ]
        file = open(os.path.join(self.output_dir, "prompts.txt"), 'r')
        prompt_content = file.read()
        file.close()
        for i in expected_prompt_fragments:
            self.assertIn(i, prompt_content)


if __name__ == "__main__":
    unittest.main()


