from string import Template

from .prompt_builder import FunctionalPrompt, System, User


class Question(User):
    def plaintext(self) -> str:
        return '\nQuestion: ' + self.content
    def openai_message(self) -> dict:
        return { "role": "user", "content": 'Question: ' + self.content }

class NoExamplesReactPrompt(FunctionalPrompt):
    def __init__(self, question, max_llm_calls):
        system_prompt = \
f"""
Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {max_llm_calls} calls to the wikipedia API.
When you look for a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result. The wikipedia pages are formatted in Markdown.
When you receive information from wikipedia always analyze it and check what useful information have you found and what else do you need.
When you know the answer call finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the thought field.
"""
        super().__init__([ System(system_prompt), Question(question) ])


class ReflectionMessageGenerator:
    def __init__(self, reflection_message, last_reflection):
        self.reflection_message = reflection_message
        self.last_reflection = last_reflection

    def generate(self, step, max_llm_calls):
        if step == max_llm_calls - 1:
            content = self.last_reflection
        else:
            template = Template(self.reflection_message)
            content = template.substitute({'step': step})
        return System(content)
