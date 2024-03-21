from string import Template
from pydantic import BaseModel, Field

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
After the first call to wikipedia you need to always reflect on the data retrieved in the previous call and fill in all the fields
related to that in the next call.
Every time you can retrieve only a small fragment of the wikipedia page, if the retrieved information looks promising - but is cut short
you can call reflection_and_read_chunk to retrieve a consecutive chunk of the page.
You can also jump to different parts of the page using the reflection_and_lookup function. In particular you can jump to 
page sections by looking up the section headers in the MarkDown syntax.
If the the lookup function return indicates that a given keyword is found in multiple places you can use the reflection_and_next
function to retrieve the next occurence of that keyword.
If a lookup does not return meaningful information you can lookup synonyms of the word you are looking for.

When you need to know a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result you don't need to call get for it.

The wikipedia pages are formatted in Markdown.
When you know the answer call reflect_and_finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the next_actions_plan field.
"""
        super().__init__([ System(system_prompt), Question(question) ])


class Reflection(BaseModel):
    how_relevant: int = Field(
        ...,
        description="Was the last retrieved information relevant for answering this question? Choose 1, 2, 3, 4, or 5. If no information was retrieved yet please choose 0"
    )
    why_relevant: str = Field(..., description="Why the retrieved information was relevant? If no information was retrieved yet please answer with empty string")
    next_actions_plan: str = Field(..., description="")


