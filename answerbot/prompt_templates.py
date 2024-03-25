from string import Template
from pydantic import BaseModel, Field

from .prompt_builder import FunctionalPrompt, System, User


class Question(User):
    def plaintext(self) -> str:
        return '\nQuestion: ' + self.content
    def openai_message(self) -> dict:
        return { "role": "user", "content": 'Question: ' + self.content }

class NoExamplesReactPrompt(FunctionalPrompt):
    def __init__(self, question, max_llm_calls, reflection_class):
        if reflection_class is not None:
            prefix = reflection_class.__name__.lower() + "_and_"
        else:
            prefix = ''
        system_prompt = \
f"""
Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {max_llm_calls - 1} calls to the wikipedia API.
After the first call to wikipedia you need to always reflect on the data retrieved in the previous call.
Every time you can retrieve only a small fragment of the wikipedia page, if the retrieved information looks promising - but is cut short
you can call {prefix}read_chunk to retrieve a consecutive chunk of the page.
You can also jump to different parts of the page using the {prefix}lookup function. In particular you can jump to 
page sections by looking up the section headers in the MarkDown syntax. It is often better to use one word lookups
because two or more words can be separated somehow or used in a different order.
If a lookup does not return meaningful information you can lookup synonyms of the word you are looking for.
If the the lookup function return indicates that a given keyword is found in multiple places you can use the {prefix}next
function to retrieve the next occurence of that keyword.

When you need to know a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result you don't need to call get for it.

The wikipedia pages are formatted in Markdown.
When you know the answer call {prefix}finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.
Always try to answer the question even if it is ambiguous, just note the necessary assumptions.
"""
        #question_analysis =  "Think about ways in which the question might be ambiguous. How could it be made more precise? Can you guess the answer without consulting wikipedia? Think step by step."
        super().__init__([ System(system_prompt), Question(question) ])

PROMPTS = {
    'NERP': NoExamplesReactPrompt
}

QUESTION_CHECKS = {
    "category_and_amb": ["What kind of question you got? Is it a Yes/No question? What should be the answer to the question?", "Think about ways in which the question might be ambiguous. How could it be made more precise?"],
    "None": [],
    "amb":  ["Think about ways in which the question might be ambiguous. How could it be made more precise?"],
    "amb_and_plan": ["Think about ways in which the question might be ambiguous. How could it be made more precise?", "Think about what information do you need to get from wikipedia to answer it. Think step by step"],
    "category": ["What kind of question you got? Is it a Yes/No question? What should be the answer to the question?"],
}



class Reflection(BaseModel):
    how_relevant: int = Field(
        ...,
        description="Was the last retrieved information relevant for answering this question? Choose 1, 2, 3, 4, or 5. If no information was retrieved yet please choose 0"
    )
    why_relevant: str = Field(..., description="Why the retrieved information was relevant? If no information was retrieved yet please answer with empty string")
    next_actions_plan: str = Field(..., description="")

class ShortReflection(BaseModel):
    reflection: str = Field(..., description="Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it? Think step by step")

REFLECTIONS = {
    'None': {},
    'Reflection': { "class": Reflection },
    'ShortReflection': { "class": ShortReflection },
    'separate': { "message": "Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it?" },
    'separate_cot': { "message": "Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it? Think step by step"}
}
