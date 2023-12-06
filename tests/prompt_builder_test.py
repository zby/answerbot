import unittest
import json

from answerbot.prompt_builder import PromptMessage, System, User, Assistant, Prompt, FunctionCall, FunctionResult, PlainTextPrompt, FunctionalPrompt

import unittest
from string import Template


class TestPromptMessages(unittest.TestCase):

    def test_basic_prompt_messages(self):
        system = System("System Test")
        self.assertEqual(system.openai_message(), {"role": "system", "content": "System Test"})
        eval_system = eval(repr(system))
        self.assertEqual(system.content, eval_system.content)
        self.assertEqual(system.role, eval_system.role)

        system = System("Message with a $placeholder", template_args={'placeholder': 'Template'})
        self.assertEqual(system.openai_message(), {"role": "system", "content": "Message with a Template"})
        eval_system = eval(repr(system))
        self.assertEqual(system.content, eval_system.content)
        self.assertEqual(system.role, eval_system.role)
        self.assertEqual(system.template_args, eval_system.template_args)

        user = User("User Test")
        self.assertEqual(user.openai_message(), {"role": "user", "content": "User Test"})

        assistant = Assistant("Assistant Test")
        self.assertEqual(assistant.openai_message(), {"role": "assistant", "content": "Assistant Test"})

    def test_function_call(self):
        function_call = FunctionCall("Test Function", reason="For testing", param1="value1", param2="value2")
        self.assertEqual(function_call.name, "Test Function")
        self.assertEqual(function_call.reason, "For testing")
        self.assertEqual(function_call.args, {"param1": "value1", "param2": "value2"})
        expected = {
            "role": "assistant",
            "content": '',
            "function_call": {
                "name": "Test Function",
                "arguments": json.dumps({"param1": "value1", "param2": "value2", "reason": "For testing"}),
            },
        }
        self.assertEqual(function_call.openai_message(), expected)

        eval_function_call = eval(repr(function_call))
        self.assertEqual(function_call.name, eval_function_call.name)
        self.assertEqual(function_call.reason, eval_function_call.reason)
        self.assertEqual(function_call.args, eval_function_call.args)

    def test_function_result(self):
        function_result = FunctionResult("Test Function", "Result")
        self.assertEqual(function_result.name, "Test Function")
        self.assertEqual(function_result.content, "Result")
        expected = {
            "role": "function",
            "name": "Test Function",
            "content": "Result",
        }
        self.assertEqual(function_result.openai_message(), expected)

        eval_function_result = eval(repr(function_result))
        self.assertEqual(function_result.name, eval_function_result.name)
        self.assertEqual(function_result.content, eval_function_result.content)



class TestPrompt(unittest.TestCase):

    def test_plaintext_prompt(self):
        system = System("System Test")
        user = User("User Test")
        assistant = Assistant("Assistant Test")
        prompt = PlainTextPrompt([system, user, assistant])
        self.assertEqual(prompt.to_text(), "System Test\nUser Test\nAssistant Test")

    def test_functional_prompt(self):
        function_call1 = FunctionCall("Func1", param1="value1")
        function_call2 = FunctionCall("Func2", param1="value1", param2="value2")
        prompt = FunctionalPrompt([function_call1, function_call2])
        expected_messages = [function_call1.openai_message(), function_call2.openai_message()]
        self.assertEqual(prompt.to_messages(), expected_messages)

    def test_Prompt_repr(self):
        system = System("System Test")
        user = User("User Test")
        assistant = Assistant("Assistant Test")
        prompt = Prompt([system, user, assistant])
        eval_prompt = eval(repr(prompt))
        self.assertEqual(len(prompt.parts), len(eval_prompt.parts))
        for part, eval_part in zip(prompt.parts, eval_prompt.parts):
            self.assertEqual(part.content, eval_part.content)
            self.assertEqual(part.role, eval_part.role)

if __name__ == "__main__":
    unittest.main()