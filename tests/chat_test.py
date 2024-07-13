import pytest
from dataclasses import dataclass

from llm_easy_tools import ToolResult

from answerbot.chat import expand_toolbox, HasLLMTools, LLMFunction, Chat, Prompt, SystemPrompt


def test_expand_toolbox():
    def simple_tool():
        return "Simple tool result"

    class MockToolProvider(HasLLMTools):
        def get_llm_tools(self):
            return [lambda: "Tool from MockToolProvider"]

    mock_llm_function = LLMFunction(lambda: "LLM Function result")

    toolbox = [simple_tool, MockToolProvider(), mock_llm_function]

    expanded_tools = expand_toolbox(toolbox)

    assert len(expanded_tools) == 3

    assert expanded_tools[0]() == "Simple tool result"
    assert expanded_tools[1]() == "Tool from MockToolProvider"
    assert expanded_tools[2]() == "LLM Function result"

    # Check that all items in the expanded toolbox are either Callable or LLMFunction
    assert all(callable(tool) or isinstance(tool, LLMFunction) for tool in expanded_tools)

@dataclass(frozen=True)
class GreetingPrompt(Prompt):
    name: str
    time_of_day: str

    @property
    def hello(self):
        return "Hello"

@dataclass(frozen=True)
class QueryPrompt(Prompt):
    question: str
    context: str

templates = {
    'GreetingPrompt': "{{hello}} {{name}}! Good {{time_of_day}}.",
    'QueryPrompt': "Question: {{question}}\nContext: {{context}}"
}

def test_chat_with_custom_prompts():
    chat = Chat(model="gpt-3.5-turbo", templates=templates)
    
    greeting = GreetingPrompt(name="Alice", time_of_day="morning")
    chat.append(greeting)
    
    query = QueryPrompt(question="What's the weather like?", context="It's summer.")
    chat.append(query)
    
    assert len(chat.messages) == 2
    assert chat.messages[0]['content'] == "Hello Alice! Good morning."
    assert chat.messages[1]['content'] == "Question: What's the weather like?\nContext: It's summer."

def test_chat_with_system_message():
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
    chat.append(tool_result)
    
    # Assert that the message was added correctly
    assert len(chat.messages) == 1
    assert chat.messages[0]['role'] == 'tool'
    assert chat.messages[0]['name'] == 'TestTool'
    assert chat.messages[0]['content'] == "This is the result of the test tool."

def test_chat_append_prompt_with_c_attribute():
    @dataclass(frozen=True)
    class InvalidPrompt(Prompt):
        c: str  # This attribute will cause a conflict

    chat = Chat(
        model="gpt-3.5-turbo",
        templates={InvalidPrompt: "This is an invalid prompt: {{c}}"}
    )

    invalid_prompt = InvalidPrompt(c="This will cause an error")

    with pytest.raises(ValueError) as excinfo:
        chat.append(invalid_prompt)

    assert str(excinfo.value) == "Prompt object cannot have an attribute named 'c' as it conflicts with the context parameter in render_prompt."

def test_chat_load_templates():
    # Create a Chat instance with templates_dirs
    chat = Chat(
        model="gpt-3.5-turbo",
        templates_dirs=["tests/data/prompts1", "tests/data/prompts2"]
    )

    # Check if the templates were loaded correctly
    assert "Prompt1" in chat.templates
    assert chat.templates["Prompt1"] == "This is Prompt1 from prompts1\nSome value {{value}}\n"
    assert "Prompt2" in chat.templates
    assert chat.templates["Prompt2"] == "This is Prompt2.\nSome value {{value}}\n"

    # Create a Chat instance with only one template directory
    chat = Chat(
        model="gpt-3.5-turbo",
        templates_dirs=["tests/data/prompts2", "tests/data/prompts1"]
    )

    # Check if the templates were loaded correctly
    assert "Prompt1" in chat.templates
    assert chat.templates["Prompt1"] == "This is Prompt1 from prompts2.\n"
    assert "Prompt2" in chat.templates
    assert chat.templates["Prompt2"] == "This is Prompt2.\nSome value {{value}}\n"

def test_chat_template_initialization():
    # Create a Chat instance with both templates_dirs and templates
    chat = Chat(
        model="gpt-3.5-turbo",
        templates_dirs=["tests/data/prompts2", "tests/data/prompts1"],
        templates={
            "Prompt1": "Overwritten template {{value}}",
            "Prompt3": "New template {{value}}"
        }
    )

    # Check if the templates were loaded and overwritten correctly
    assert chat.templates["Prompt1"] == "Overwritten template {{value}}"
    assert chat.templates["Prompt2"] == "This is Prompt2.\nSome value {{value}}\n"
    assert chat.templates["Prompt3"] == "New template {{value}}"

    # Create Prompt classes that match the templates
    @dataclass(frozen=True)
    class Prompt1(Prompt):
        value: str

    @dataclass(frozen=True)
    class Prompt2(Prompt):
        value: str

    @dataclass(frozen=True)
    class Prompt3(Prompt):
        value: str

    # Test rendering the templates
    prompt1 = Prompt1(value="test1")
    message1 = chat.make_message(prompt1)

    assert message1['role'] == 'user'
    assert message1['content'] == "Overwritten template test1"

    prompt2 = Prompt2(value="test2")
    message2 = chat.make_message(prompt2)

    assert message2['role'] == 'user'
    assert message2['content'] == "This is Prompt2.\nSome value test2"

    prompt3 = Prompt3(value="test3")
    message3 = chat.make_message(prompt3)

    assert message3['role'] == 'user'
    assert message3['content'] == "New template test3"

    # Test that a template not present in either source raises a KeyError
    @dataclass(frozen=True)
    class NonexistentPrompt(Prompt):
        value: str

    with pytest.raises(KeyError):
        chat.make_message(NonexistentPrompt(value="test"))