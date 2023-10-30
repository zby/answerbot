import json
import unittest
from unittest.mock import patch


from react_prompt import ReactPrompt, FunctionalReactPrompt, TextReactPrompt


class TestReactPrompt(unittest.TestCase):

    def test_functional_react_prompt_initialization(self):
        frprompt = FunctionalReactPrompt("Bla bla bla", 300)
        self.assertIn("Bla bla bla", frprompt.to_text())
        self.assertIn("For the Action step you can call the available functions.", frprompt.to_text())

    def test_text_react_prompt_initialization(self):
        trprompt = TextReactPrompt("Bla bla bla", 300)
        self.assertIn("Bla bla bla", trprompt.to_text())
        self.assertIn("After each observation, provide the next Thought and next Action. Here are some examples:",
                      trprompt.to_text())

    def test_react_prompt_examples_chunk_size(self):
        frprompt_small = FunctionalReactPrompt("Test chunk size", 150)
        frprompt_large = FunctionalReactPrompt("Test chunk size", 450)

        # This is a simple test to check if the larger chunk size results in a longer text.
        # You may need a more precise test depending on your actual implementation.
        self.assertTrue(len(frprompt_large.to_text()) >= len(frprompt_small.to_text()))



class TestFunctionalReactPrompt(unittest.TestCase):

    def setUp(self):
        with patch.object(ReactPrompt, 'get_examples', return_value=[]):
            self.instance = FunctionalReactPrompt(question="Some question")

    def test_function_call_from_response(self):
        response = {
            'role': 'assistant',
            "content": 'bla bla bla',
            'function_call': {
                "name": "search",
                "arguments": json.dumps({"query": "Milhouse Simpson", "thought": "Looking for Milhouse details"})
            }
        }
        expected_args = json.dumps({ 'query': 'Milhouse Simpson', 'thought': "Looking for Milhouse details"})
        output = self.instance.function_call_from_response(response)
        self.assertEqual(output['name'], 'search')
        self.assertEqual(output['arguments'], expected_args)


class TestTextReactPrompt(unittest.TestCase):

    def setUp(self):
        with patch.object(ReactPrompt, 'get_examples', return_value=[]):
            self.instance = TextReactPrompt(question="Another question")

    def test_function_call_from_response_finish(self):
        response = {
            "content": "Thought: Final thought\nAction: finish[Correct Answer]"
        }
        expected_output = {
            "name": "finish",
            "arguments": json.dumps({"answer": "Correct Answer", "thought": "Final thought"})
        }
        self.assertEqual(self.instance.function_call_from_response(response), expected_output)

    def test_function_call_from_response_search(self):
        response = {
            "content": "Thought: Looking for data\nAction: search[Milhouse Simpson]"
        }
        expected_output = {
            "name": "search",
            "arguments": json.dumps({"query": "Milhouse Simpson", "thought": "Looking for data"})
        }
        self.assertEqual(self.instance.function_call_from_response(response), expected_output)



if __name__ == '__main__':
    unittest.main()
