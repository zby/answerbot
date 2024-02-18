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
When you need to know a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result you don't need to call get for it.
When a page is retrieved only a part of it is displayed, you can jump to different parts of the page by using the lookup function.
If the the lookup function return indicates that a given keyword is found in multiple places you can use the next function to retrieve the
next occurence of that keyword.
If a lookup does not return meaningful information you can lookup synonyms of the word you are looking for.
The wikipedia pages are formatted in Markdown.
When you receive information from wikipedia always analyze it and check what useful information have you found and what else do you need.
When you know the answer call finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the thought field.
"""
        super().__init__([ System(system_prompt), Question(question) ])


class ReflectionMessageGenerator:
    def __init__(self):
        self.reflection_message = "Reflect on the received information and plan next steps. This was a call to the Wikiepdia API number $step."
        self.last_reflection = "In the next call you need to formulate an answer - please reflect on the received information."

    def generate(self, step, max_llm_calls):
        if step == max_llm_calls - 1:
            content = self.last_reflection
        else:
            template = Template(self.reflection_message)
            content = template.substitute({'step': step})
        return System(content)
