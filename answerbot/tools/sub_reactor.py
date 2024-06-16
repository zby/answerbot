from typing import Any, Optional
from answerbot.react import LLMReactor, Trace, SubReactorResult


class SubReactorTool:
    def __init__(self, reactors_dict: dict[str, Any], previous_answers: Optional[dict[str, str]] = None):
        self.reactors_dict = reactors_dict
        self.previous_answers = previous_answers if previous_answers is not None else {}
#        delegate_doc_string = "Delegate the question to another researcher. There are following researchers available:\n"
#        for reactor_name in self.reactors_dict:
#            delegate_doc_string += f"{reactor_name}\n"
#        setattr(SubReactorTool.delegate, 'LLMEasyTools_description', )

    def delegate(self, question: str) -> str:
        """
        Delegate the question to a wikipedia expert.
        """
        print(f"Delegating to a wikipedia researcher the following question: {question}")
        reactor_name = 'wikipedia researcher'
        if reactor_name in self.reactors_dict:
            args = self.reactors_dict[reactor_name]
            args['question'] = question
            reactor = LLMReactor.create_reactor(**args)
            if self.previous_answers:
                previous_answers = 'What have we learned so far:\n\n'
                for question, answer in self.previous_answers.items():
                    previous_answers += f"Question: {question}\nAnswer: {answer}\n\n"
                message = {'role': 'user', 'content': previous_answers}
                reactor.trace.append(message)
            reactor.process()
            self.previous_answers[question] = str(reactor.answer)
            reflection_prompt = """So far we have researched the following questions:
{previous_answers}

What would you do next? 
If you have found enough information to answer the question you can call finish.
Otherwise you can delegate the question to another researcher.
Please don't delegate questions that we have already tried.
List five possible questions to the next researcher.
Then specify which one of them you would like to do next.
Please explain your decision.
"""
            result = SubReactorResult(reactor.generate_report(), reactor.trace, reflection_prompt)
            print(f"SubReactor result: {str(result)}")
            return result
        else:
            raise ValueError(f"No researcher found for the name: {reactor_name}")
