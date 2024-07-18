import pytest
from dataclasses import dataclass

from llm_easy_tools import ToolResult, LLMFunction
from answerbot.chat import Chat, Prompt, SystemPrompt
from jinja2 import Environment, DictLoader
from jinja2.ext import Extension

def test_append():
    @dataclass(frozen=True)
    class GreetingPrompt(Prompt):
        name: str
        time_of_day: str

        @property
        def hello(self):
            return "Hello"

    templates = {
        'GreetingPrompt': "{{hello}} {{name}}! Good {{time_of_day}}.",
    }
    chat = Chat(model="gpt-3.5-turbo", templates=templates)
    
    greeting = GreetingPrompt(name="Alice", time_of_day="morning")
    chat.append(greeting)
    
    assert len(chat.messages) == 1
    assert chat.messages[0]['content'] == "Hello Alice! Good morning."

    chat.append("Hello, can you help me?")
    assert len(chat.messages) == 2
    assert chat.messages[1]['content'] == "Hello, can you help me?"

def test_system_prompt():
    chat = Chat(
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful AI assistant."
    )

    assert len(chat.messages) == 1
    assert chat.messages[0]['role'] == 'system'
    assert chat.messages[0]['content'] == "You are a helpful AI assistant."

    @dataclass(frozen=True)
    class SpecialSystemPrompt(SystemPrompt):
        capabilities: str

    templates = {
        'SpecialSystemPrompt': "You are a helpful AI assistant. Your capabilities include: {{capabilities}}."
    }

    system_prompt = SpecialSystemPrompt(
        capabilities="answering questions, providing information, and assisting with tasks"
    )

    chat = Chat(
        model="gpt-3.5-turbo",
        templates=templates,
        system_prompt=system_prompt
    )

    assert len(chat.messages) == 1
    assert chat.messages[0]['role'] == 'system'
    assert chat.messages[0]['content'] == (
        "You are a helpful AI assistant. Your capabilities include: "
        "answering questions, providing information, and assisting with tasks."
    )

    # Test adding a user message after system prompt
    chat.append({"role": "user", "content": "Hello, can you help me?"})

    assert len(chat.messages) == 2
    assert chat.messages[1]['role'] == 'user'
    assert chat.messages[1]['content'] == "Hello, can you help me?"

def test_chat_append_tool_result():
    chat = Chat(model="gpt-3.5-turbo")
    
    # Create a ToolResult
    tool_result = ToolResult(
        tool_call_id="123",
        name="TestTool",
        output="This is the result of the test tool.",
    )
    
    # Append the ToolResult to the chat
    chat.append(tool_result.to_message())
    
    # Assert that the message was added correctly
    assert len(chat.messages) == 1
    assert chat.messages[0]['role'] == 'tool'
    assert chat.messages[0]['name'] == 'TestTool'
    assert chat.messages[0]['content'] == "This is the result of the test tool."

def test_template_loading():
    # Create a Chat instance with templates_dirs
    chat = Chat(
        model="gpt-3.5-turbo",
        templates_dirs=["tests/data/prompts1", "tests/data/prompts2"]
    )

    # Check if the templates were loaded correctly
    t = chat.template_env.get_template("Prompt1")
    assert t.render({"value": "test"}) == 'This is Prompt1 from prompts1\nSome value: "test"'
    t = chat.template_env.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Create a Chat instance with prompts2 first
    chat = Chat(
        model="gpt-3.5-turbo",
        templates_dirs=["tests/data/prompts2", "tests/data/prompts1"]
    )

    # Check if the templates were loaded correctly
    t = chat.template_env.get_template("Prompt1")
    assert t.render({}) == "This is Prompt1 from prompts2."
    t = chat.template_env.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

def test_renderer_with_templates_dict():
    # Create a Chat instance with templates dictionary and a template directory
    chat = Chat(
        model="gpt-3.5-turbo",
        templates={
            "CustomPrompt1": "This is a custom prompt: {{value}}",
            "CustomPrompt2": "Another custom prompt: {{name}}",
            "Prompt1": "Overridden Prompt1: {{value}}"  # This should override the one from disk
        },
        templates_dirs=["tests/data/prompts1"]
    )

    # Check if the templates were loaded correctly
    t1 = chat.template_env.get_template("CustomPrompt1")
    assert t1.render({"value": "test"}) == 'This is a custom prompt: test'

    t2 = chat.template_env.get_template("CustomPrompt2")
    assert t2.render({"name": "John"}) == 'Another custom prompt: John'

    # Test that the template from disk is loaded
    t3 = chat.template_env.get_template("Prompt2")
    assert t3.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Test that Prompt1 is overridden
    t4 = chat.template_env.get_template("Prompt1")
    assert t4.render({"value": "test"}) == 'Overridden Prompt1: test'

def test_chat_with_custom_environment():
    # Create a custom Jinja2 Environment
    custom_templates = {
        "CustomPrompt": "This is a custom prompt with {{value}}",
    }
    custom_env = Environment(loader=DictLoader(custom_templates))

    # Create a Chat instance with the custom environment
    chat = Chat(
        model="gpt-3.5-turbo",
        template_env=custom_env
    )

    # Test if the custom template is accessible
    t = chat.template_env.get_template("CustomPrompt")
    assert t.render({"value": "test"}) == 'This is a custom prompt with test'

    # Ensure that passing both template_env and templates/templates_dirs raises an error
    with pytest.raises(ValueError):
        Chat(
            model="gpt-3.5-turbo",
            template_env=custom_env,
            templates={"AnotherPrompt": "This should raise an error"}
        )

class CustomExtension(Extension):
    def __init__(self, environment):
        super().__init__(environment)
        environment.filters['custom_upper'] = lambda x: x.upper()

def test_chat_with_environment_extension():
    # Create a custom Jinja2 Environment with an extension
    custom_templates = {
        "UppercasePrompt": "This prompt uses a custom filter: {{ value | custom_upper }}",
    }
    custom_env = Environment(
        loader=DictLoader(custom_templates),
        extensions=[CustomExtension]
    )

    # Create a Chat instance with the custom environment
    chat = Chat(
        model="gpt-3.5-turbo",
        template_env=custom_env
    )

    # Test if the custom template with the extension is working
    t = chat.template_env.get_template("UppercasePrompt")
    assert t.render({"value": "test"}) == 'This prompt uses a custom filter: TEST'

    # Ensure that the custom filter is available in render_prompt
    @dataclass(frozen=True)
    class UppercasePrompt(Prompt):
        value: str

    prompt = UppercasePrompt(value="hello")
    rendered = chat.render_prompt(prompt)
    assert rendered == 'This prompt uses a custom filter: HELLO'

def test_render_prompt():
    chat = Chat(
        model="gpt-3.5-turbo",
        templates={
            "TwoValsPrompt": "value1: {{value1}}\nvalue2: {{value2}}"
        },
    )
    @dataclass(frozen=True)
    class TwoValsPrompt(Prompt):
        value1: str

    prompt1 = TwoValsPrompt(value1="test1")

    assert chat.render_prompt(prompt1) == 'value1: test1\nvalue2: '

    # Check kwargs
    assert chat.render_prompt(prompt1, value2="test2") == 'value1: test1\nvalue2: test2'