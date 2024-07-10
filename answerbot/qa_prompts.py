from dataclasses import dataclass
from answerbot.chat import Prompt, SystemPrompt


@dataclass(frozen=True)
class Question(Prompt):
    question: str
    max_llm_calls: int

@dataclass
class Answer:
    """
    Answer to the question.
    """
    answer: str
    reasoning: str

@dataclass(frozen=True)
class StepInfo(Prompt):
    step: int
    max_steps: int

# dictionary for prompt templates
wiki_researcher_prompts = {
    SystemPrompt: """You are a helpful assistant with extensive knowledge of wikipedia.
You always try to support your answer with quotes from wikipedia.
You remember that the information you receive from the wikipedia api is not the full page - it is just a fragment.
You always try to answer the user question, even if it is ambiguous, just note the necessary assumptions.
You Work carefully - never make two calls to wikipedia in the same step.""",

    Question: """Please answer the following question. You can use wikipedia for reference - but think carefully about what pages exist at wikipedia.
You have only {{max_llm_calls}} calls to the wikipedia API.
When searching wikipedia never make any complex queries, always decide what is the main topic you are searching for and put it in the search query.
When you want to know a property of an object or person - first find the page of that object or person and then browse it to find the property you need.

When you know the answer call Answer. Please make the answer as short as possible. If it can be answered with yes or no that is best.
Remove all explanations from the answer and put them into the reasoning field.

Question: {{question}}""",

    Answer: """The answer to the question:"{{c.question}}" is:
{{ answer }}

Reasoning:
{{ reasoning }}""",

    StepInfo: """
Step: {{step + 1}} of {{max_steps + 1}}
{% if step >= max_steps - 1 %}
This was the last data retrieval in the next step you must provide an answer to the user question
{% endif %}
"""
}

main_researcher_prompts = {
    SystemPrompt: """
You are to take a role of a main researcher dividing work and delegating tasks to your assistants.
You always try to make the delegated tasks as simple as possible.
Always try to answer the question, even if it is ambiguous, just note the necessary assumptions.""",

    Question: """
Please answer the users question.

You need to carefully divide the work into tasks that would require the least amount of calls to the wikipedia API,
and then delegate them to the assistant.
The questions you ask the assistant need to be as simple and specific as possible and it should be a grammatical question.
You can call finish when you think you have enough information to answer the question.
You can delegate only {{max_llm_calls}} tasks to the assistant.

Question: {{question}}"""
}

main_researcher_prompts = wiki_researcher_prompts | main_researcher_prompts