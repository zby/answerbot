from string import Template
from pydantic import BaseModel, Field


def aae_sys_prompt(max_llm_calls, prefix=None):
        return f'''
You are to take a role of a specialist in the new 
EU artificial intelligence act. Please answer the 
following question. You can use a EU artificial intelligence
act website to find the information. 
You can use {prefix}search_aae to make a search. You have only
{max_llm_calls} search attempts. Every time you do a search,
you will be given a list of articles where the search query 
is found: url, title, and a short excerpt. 

You can then use {prefix}goto_url to go to a found article that
you think is relevant. 

After you go to a page you need to always reflect on the data retrieved in the previous call.
you can call {prefix}read_chunk to retrieve a consecutive chunk of the page.
The pages are formatted in Markdown.

When you know the answer call {prefix}finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.
Always try to answer the question even if it is ambiguous, just note the necessary assumptions.
'''


def zero_shot_prompt(max_llm_calls, prefix=None):
      return\
f"""
Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {max_llm_calls - 1} calls to the wikipedia API.
After the first call to wikipedia you need to always reflect on the data retrieved in the previous call.
To retrieve the first document you need to call search.

When you need to know a property of something or someone - search for that something page instead of using that property in the search.
The search function automatically retrieves the first search result you don't need to call get for it.

The wikipedia pages are formatted in Markdown.
When you know the answer call finish. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.
Always try to answer the question even if it is ambiguous, just note the necessary assumptions.
"""

PROMPTS = {
    'NERP': zero_shot_prompt,
    'AAE': aae_sys_prompt,
}

QUESTION_CHECKS = {
    "category_and_amb": ["What kind of question you got? Is it a Yes/No question? What should be the answer to the question?", "Think about ways in which the question might be ambiguous. How could it be made more precise?"],
    "None": [],
    "amb":  ["Think about ways in which the question might be ambiguous. How could it be made more precise?"],
    "amb_and_plan": ["Think about ways in which the question might be ambiguous. How could it be made more precise?", "Think about what information do you need to get from wikipedia to answer it. Think step by step"],
    "category": ["What kind of question you got? Is it a Yes/No question? What should be the answer to the question?"],
}


class SimpleReflection(BaseModel):
    is_relevant: bool = Field(..., description='True if you found any useful information in the last response')


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
    'Reflection': { "reflection_class": Reflection, 'detached': False },
    'SimpleReflection': {'reflection_class': SimpleReflection, 'detached': False},
    'ShortReflection': { "reflection_class": ShortReflection, 'detached': False },
    'ReflectionDetached': {'reflection_class': Reflection},
    'ShortReflectionDetached': {'reflection_class': ShortReflection},
    'separate': { "message": "Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it?" },
    'separate_cot': { "message": "Reflect on the information you have gathered so far. Was the last retrieved information relevant for answering the question? What additional information you need, why and how you can get it? Think step by step"}
}
