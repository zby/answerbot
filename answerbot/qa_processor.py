from dataclasses import dataclass
from answerbot.chat import Chat, Prompt, HasLLMTools, expand_toolbox, render_prompt
from answerbot.tools.wiki_tool import WikipediaTool 
from typing import Callable

from llm_easy_tools import LLMFunction

import logging
import litellm
from dotenv import load_dotenv


# Configure logging for this module
logger = logging.getLogger('qa_processor')


@dataclass(frozen=True)
class SystemPrompt(Prompt):
    """
    System prompt for the chat.
    """
    pass

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

# New dictionary for prompt templates
prompt_templates = {
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

    Answer: """The answer to the question:"{{context.question}}" is:
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

@dataclass
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[type, str]

    def __post_init__(self):
        self.step = 0
        self.chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            system_prompt=SystemPrompt(),
            context=self,
            templates=self.prompt_templates
        )

    def get_tools(self) -> list[Callable|LLMFunction]:
        tools = [Answer]
        if self.step < self.max_iterations:
            tools.extend(expand_toolbox(self.toolbox))
        return tools

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = self.chat
        chat.entries.append(Question(question, self.max_iterations))
        while(self.step <= self.max_iterations):
            tools = self.get_tools()
            self.chat.process(self, tools)
            result = chat.entries[-1]
            if result.error:
                raise Exception(result.error)
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logger.warning(soft_error)
            if isinstance(result.output, Answer):
                answer = result.output
                return render_prompt(prompt_templates[Answer], answer, {'question': question})
            chat.entries.append(StepInfo(self.step, self.max_iterations))
            self.step += 1
        return None

if __name__ == "__main__":

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    #litellm.success_callback=["helicone"]
    #litellm.set_verbose=True

    #model='claude-3-5-sonnet-20240620'
    model="claude-3-haiku-20240307"
    #model='gpt-3.5-turbo'
    question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

    app = QAProcessor(
        toolbox=[WikipediaTool(chunk_size=400)],
        max_iterations=5,
        model=model,
        prompt_templates=prompt_templates
    )

    #answer = Answer(answer="Something", reasoning="Because")
    #print(render_prompt(Answer.template(), answer, app))
    #print(app.question)

    print(app.process(question))