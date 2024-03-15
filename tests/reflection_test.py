from answerbot.react import Reflection


def test_validation():
    tool_args = {
        'how_relevant': '5',
        'link': 'Kiss and Tell (1945 film)',
        'next_actions_plan': 'Retrieve the information about the woman who portrayed '
                             "Corliss Archer in the film 'Kiss and Tell'.",
        'why_relevant': "The page 'Corliss Archer' contains information about the "
                        "woman who portrayed Corliss Archer in the film 'Kiss and "
                        "Tell'."
    }

    param_class = Reflection
    param = param_class(**tool_args)
    assert param.how_relevant == 5 # str converted to int


