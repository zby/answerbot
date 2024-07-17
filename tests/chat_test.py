import pytest
from dataclasses import dataclass

from llm_easy_tools import ToolResult, LLMFunction
from answerbot.chat import Chat, Prompt, SystemPrompt, Jinja2Renderer



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

def test_append_with_system_message():
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
    # Create a TemplateManager instance with templates_dirs
    renderer = Jinja2Renderer(
        templates_dirs=["tests/data/prompts1", "tests/data/prompts2"]
    )

    # Check if the templates were loaded correctly
    t = renderer.env.get_template("Prompt1")
    assert t.render({"value": "test"}) == 'This is Prompt1 from prompts1\nSome value: "test"'
    t = renderer.env.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Create a TemplateManager instance with prompts2 first
    renderer = Jinja2Renderer(
        templates_dirs=["tests/data/prompts2", "tests/data/prompts1"]
    )

    # Check if the templates were loaded correctly
    t = renderer.env.get_template("Prompt1")
    assert t.render({}) == "This is Prompt1 from prompts2."
    t = renderer.env.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

def test_render_prompt():
    renderer = Jinja2Renderer(
        templates={
            "TwoValsPrompt": "value1: {{value1}}\nvalue2: {{value2}}"
        },
    )
    @dataclass(frozen=True)
    class TwoValsPrompt(Prompt):
        value1: str

    prompt1 = TwoValsPrompt(value1="test1")

    assert renderer.render_prompt(prompt1) == 'value1: test1\nvalue2: '

    # Check kwargs
    assert renderer.render_prompt(prompt1, value2="test2") == 'value1: test1\nvalue2: test2'

def test_renderer_with_templates_dict():
    # Create a TemplateManager instance with templates dictionary and a template directory
    renderer = Jinja2Renderer(
        templates={
            "CustomPrompt1": "This is a custom prompt: {{value}}",
            "CustomPrompt2": "Another custom prompt: {{name}}",
            "Prompt1": "Overridden Prompt1: {{value}}"  # This should override the one from disk
        },
        templates_dirs=["tests/data/prompts1"]
    )

    # Check if the templates were loaded correctly
    t1 = renderer.env.get_template("CustomPrompt1")
    assert t1.render({"value": "test"}) == 'This is a custom prompt: test'

    t2 = renderer.env.get_template("CustomPrompt2")
    assert t2.render({"name": "John"}) == 'Another custom prompt: John'

    # Test that the template from disk is loaded
    t3 = renderer.env.get_template("Prompt2")
    assert t3.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Test that Prompt1 is overridden
    t4 = renderer.env.get_template("Prompt1")
    assert t4.render({"value": "test"}) == 'Overridden Prompt1: test'

