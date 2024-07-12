from dataclasses import dataclass
from typing import Callable, Any, Optional

from llm_easy_tools import LLMFunction, get_tool_defs

import logging

from answerbot.chat import Chat, HasLLMTools, SystemPrompt, render_prompt, expand_toolbox
from answerbot.tools.observation import Observation
from answerbot.reflector import reflect, plan_next_action 
from answerbot.knowledgebase import KnowledgeBase
from answerbot.qa_prompts import Question, Answer, StepInfo

# Configure logging for this module
logger = logging.getLogger('qa_processor')


@dataclass(frozen=True)
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[type, str]
    name: Optional[str] = None

    def get_tools(self, step: int) -> list[Callable|LLMFunction|HasLLMTools]:
        if step < self.max_iterations:
            return[Answer, *self.toolbox]
        else:
            return [Answer]

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            system_prompt=SystemPrompt(),
            context=self,
            templates=self.prompt_templates,
        )
        if self.name:
            chat.metadata = {"tags": [self.name]}
        chat.append(Question(question, self.max_iterations))

        what_have_we_learned = KnowledgeBase()

        observation = None
        reflection_string = None
        for step in range(self.max_iterations + 1):
            chat.append(StepInfo(step, self.max_iterations))
            tools = self.get_tools(step)
            results = chat.process(tools)
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    answer = output
                    logger.info(f"Answer: '{answer}' for question: '{question}'")
                    return render_prompt(self.prompt_templates[Answer], answer, {'question': question})
                observation = output
                if isinstance(output, Observation):
                    if observation.reflection_needed():
                        reflection_string = reflect(self.model, self.prompt_templates, question, observation, what_have_we_learned)
                available_tools = self.get_tools(step)
                planning_string = plan_next_action(self.model, self.prompt_templates, question, available_tools, observation, reflection_string)
                chat.append({'role': 'user', 'content': planning_string})
            logger.info(f"Step: {step} for question: '{question}'")

        return None


@dataclass(frozen=True)
class QAProcessorDeep(QAProcessor):
    sub_processor_config: Optional[dict[str, Any]] = None
    delegate_description: Optional[str] = None

    def __post_init__(self):
        if self.sub_processor_config:
            if self.toolbox:
                raise(Exception("Cannot set toolbox when using sub processor"))
            if self.delegate_description is None:
                raise(Exception("Must set deletage_description when usign sub processor"))

            sub_processor =  QAProcessor(**self.sub_processor_config)

            def delegate(sub_question: str):
                logger.info(f"{self.delegate_description}: '{sub_question}'")
                return sub_processor.process(sub_question)

            delegate_fun = LLMFunction(delegate, description=self.delegate_description)

            object.__setattr__(self, 'toolbox', [delegate_fun])
