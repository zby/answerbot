from answerbot.qa_processor import QAProcessor, Answer, HasLLMTools, expand_toolbox, ReflectionPrompt, PlanningPrompt
from answerbot.tools.observation import History, Observation, KnowledgePiece
from llm_easy_tools import ToolResult, LLMFunction

import pytest

@pytest.fixture
def sample_history():
    history = History()

    # Create and add observations
    observation1 = Observation(
        content="The weather in Tokyo is sunny with a high of 25°C.",
        operation="weather_lookup",
        source="https://weather.example.com/tokyo",
        quotable=True,
        goal="Check Tokyo weather"
    )
    observation2 = Observation(
        content="Tokyo is the capital city of Japan, located on the eastern coast of Honshu island.",
        operation="city_info",
        source="https://geography.example.com/tokyo",
        quotable=True,
        goal="Get information about Tokyo"
    )

    history.add_observation(observation1)
    history.add_observation(observation2)

    # Create and add knowledge pieces
    knowledge_piece1 = KnowledgePiece(
        source=observation1,
        content="The current weather in Tokyo is sunny.",
        quotes=["The weather in Tokyo is sunny with a high of 25°C."]
    )
    knowledge_piece2 = KnowledgePiece(
        source=observation2,
        content="Tokyo is the capital of Japan.",
        quotes=["Tokyo is the capital city of Japan, located on the eastern coast of Honshu island."]
    )

    history.add_knowledge_piece(knowledge_piece1)
    history.add_knowledge_piece(knowledge_piece2)

    return history


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


def test_reflection_prompt(sample_history):
    question = "What is the weather in Tokyo?"
    reflection_prompt = ReflectionPrompt(history=sample_history, question=question)

    qa = QAProcessor(model='aaa', toolbox=[], max_iterations=1)
    chat = qa.make_chat()
    result = chat.render_prompt(reflection_prompt)
    assert 'Tokyo' in result

def test_planning_prompt(sample_history):
    question = "What is the weather in Tokyo?"
    planning_prompt = PlanningPrompt(history=sample_history, question=question, available_tools=[])

    qa = QAProcessor(model='aaa', toolbox=[], max_iterations=1, prompt_templates_dirs=['answerbot/templates/wiki_researcher'])
    chat = qa.make_chat()
    result = chat.render_prompt(planning_prompt)
    assert 'Tokyo' in result