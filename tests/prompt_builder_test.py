import json
from answerbot.prompt_builder import PromptMessage, System, User, Assistant, Prompt, FunctionCall, FunctionResult, PlainTextPrompt, FunctionalPrompt

def test_basic_prompt_messages():
    system = System("System Test")
    assert system.openai_message() == {"role": "system", "content": "System Test"}
    eval_system = eval(repr(system))
    assert system.content == eval_system.content
    assert system.role == eval_system.role

    system = System("Message with a $placeholder", template_args={'placeholder': 'Template'})
    assert system.openai_message() == {"role": "system", "content": "Message with a Template"}
    eval_system = eval(repr(system))
    assert system.content == eval_system.content
    assert system.role == eval_system.role
    assert system.template_args == eval_system.template_args

    user = User("User Test")
    assert user.openai_message() == {"role": "user", "content": "User Test"}

    assistant = Assistant("Assistant Test")
    assert assistant.openai_message() == {"role": "assistant", "content": "Assistant Test"}

def test_function_call():
    function_call = FunctionCall("Test Function", reason="For testing", param1="value1", param2="value2")
    assert function_call.name == "Test Function"
    assert function_call.args == {"reason": "For testing", "param1": "value1", "param2": "value2"}
    expected = {
        "role": "assistant",
        "content": '',
        "function_call": {
            "name": "Test Function",
            "arguments": json.dumps({"reason": "For testing", "param1": "value1", "param2": "value2"}),
        },
    }
    assert function_call.openai_message() == expected

    eval_function_call = eval(repr(function_call))
    assert function_call.name == eval_function_call.name
    assert function_call.args == eval_function_call.args

def test_function_result():
    function_result = FunctionResult("Test Function", "Result")
    assert function_result.name == "Test Function"
    assert function_result.content == "Result"
    expected = {
        "role": "function",
        "name": "Test Function",
        "content": "Result",
    }
    assert function_result.openai_message() == expected

    eval_function_result = eval(repr(function_result))
    assert function_result.name == eval_function_result.name
    assert function_result.content == eval_function_result.content

def test_plaintext_prompt():
    system = System("System Test")
    user = User("User Test")
    assistant = Assistant("Assistant Test")
    prompt = PlainTextPrompt([system, user, assistant])
    assert prompt.to_text() == "System Test\nUser Test\nAssistant Test"

def test_functional_prompt():
    function_call1 = FunctionCall("Func1", param1="value1")
    function_call2 = FunctionCall("Func2", param1="value1", param2="value2")
    prompt = FunctionalPrompt([function_call1, function_call2])
    expected_messages = [function_call1.openai_message(), function_call2.openai_message()]
    assert prompt.to_messages() == expected_messages

def test_Prompt_repr():
    system = System("System Test")
    user = User("User Test")
    assistant = Assistant("Assistant Test")
    prompt = Prompt([system, user, assistant])
    eval_prompt = eval(repr(prompt))
    assert len(prompt.parts) == len(eval_prompt.parts)
    for part, eval_part in zip(prompt.parts, eval_prompt.parts):
        assert part.content == eval_part.content
        assert part.role == eval_part.role
