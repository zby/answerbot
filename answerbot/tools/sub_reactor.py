from typing import Any
from answerbot.react import LLMReactor

def aaa():
    return "aaa"

class SubReactorTool:
    def __init__(self, reactors_dict: dict[str, Any]):
        self.reactors_dict = reactors_dict
#        delegate_doc_string = "Delegate the question to another researcher. There are following researchers available:\n"
#        for reactor_name in self.reactors_dict:
#            delegate_doc_string += f"{reactor_name}\n"
#        setattr(SubReactorTool.delegate, 'LLMEasyTools_description', )

    def delegate(self, question: str) -> str:
        """
        Delegate the question to a wikipedia expert.
        """
        reactor_name = 'wikipedia researcher'
        if reactor_name in self.reactors_dict:
            args = self.reactors_dict[reactor_name]
            args['question'] = question
            reactor = LLMReactor.create_reactor(**args)
            reactor.process()
            return reactor.generate_report()
        else:
            raise ValueError(f"No researcher found for the name: {reactor_name}")
