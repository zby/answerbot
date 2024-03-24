from answerbot.prompt_templates import NoExamplesReactPrompt, Reflection, ShortReflection, QUESTION_CHECKS

def test_prompt():
    question = "What is your name?"
    max_llm_calls = 5
    reflection_class = ShortReflection
    prompt = NoExamplesReactPrompt(question, max_llm_calls, reflection_class)
    system_message = prompt.to_messages()[0]
    assert "call shortreflection_and_read_chunk to retrieve" in system_message['content']

    reflection_class = None
    prompt = NoExamplesReactPrompt(question, max_llm_calls, reflection_class)
    system_message = prompt.to_messages()[0]
    assert "call read_chunk to retrieve" in system_message['content']

