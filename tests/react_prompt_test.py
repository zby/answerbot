from answerbot.react_prompt import ReactPrompt, FunctionalReactPrompt, TextReactPrompt

def test_functional_react_prompt_initialization():
    frprompt = FunctionalReactPrompt("Bla bla bla", 300)
    assert "Bla bla bla" in str(frprompt)
    assert "For the Action step you can call the available functions." in str(frprompt)

def test_text_react_prompt_initialization():
    trprompt = TextReactPrompt("Bla bla bla", 300)
    assert "Bla bla bla" in trprompt.to_text()
    assert "After each observation, provide the next Thought and next Action. Here are some examples:" in trprompt.to_text()

def test_react_prompt_examples_chunk_size():
    frprompt_small = FunctionalReactPrompt("Test chunk size", 150)
    frprompt_large = FunctionalReactPrompt("Test chunk size", 450)

    assert len(str(frprompt_large)) >= len(str(frprompt_small))
