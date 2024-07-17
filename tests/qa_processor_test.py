from answerbot.qa_processor import QAProcessor, Answer, HasLLMTools, expand_toolbox
from llm_easy_tools import ToolResult, LLMFunction


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

 