from dataclasses import dataclass
from typing import Callable

from llm_easy_tools import LLMFunction

import logging
import litellm
from dotenv import load_dotenv

from answerbot.chat import Chat, HasLLMTools, expand_toolbox, render_prompt
from answerbot.tools.wiki_tool import WikipediaTool
from answerbot.tools.observation import Observation
from answerbot.reflector import reflect, plan_next_action 
from answerbot.clean_reflection import KnowledgeBase
from answerbot.qa_prompts import Question, Answer, StepInfo, SystemPrompt, prompt_templates

# Configure logging for this module
logger = logging.getLogger('qa_processor')


@dataclass(frozen=True)
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[type, str]

    def get_tools(self, step: int) -> list[Callable|LLMFunction]:
        tools = [Answer]
        if step < self.max_iterations:
            tools.extend(expand_toolbox(self.toolbox))
        return tools

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            system_prompt=SystemPrompt(),
            context=self,
            templates=self.prompt_templates
        )
        chat.entries.append(Question(question, self.max_iterations))

        what_have_we_learned = KnowledgeBase()

        for step in range(self.max_iterations + 1):
            logger.info(f"Step: {step} for question: '{question}'")
            tools = self.get_tools(step)
            output = chat.process(tools)[0]
            if isinstance(output, Answer):
                answer = output
                logger.info(f"Answer: '{answer}' for question: '{question}'")
                return render_prompt(prompt_templates[Answer], answer, {'question': question})
            chat.entries.append(StepInfo(step, self.max_iterations))
            if isinstance(output, Observation) and output.reflection_needed():
                observation = output
                reflection_string = reflect(self.model, observation, question, what_have_we_learned)

                planning_string = plan_next_action(self.model, observation, question, reflection_string) 
                chat.entries.append({'role': 'user', 'content': planning_string})
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