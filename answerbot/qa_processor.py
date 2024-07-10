from dataclasses import dataclass
from typing import Callable, Any

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
            templates=self.prompt_templates
        )
        chat.append(Question(question, self.max_iterations))

        what_have_we_learned = KnowledgeBase()

        for step in range(self.max_iterations + 1):
            logger.info(f"Step: {step} for question: '{question}'")
            tools = self.get_tools(step)
            output = chat.process(tools)[0]
            if isinstance(output, Answer):
                answer = output
                logger.info(f"Answer: '{answer}' for question: '{question}'")
                return render_prompt(self.prompt_templates[Answer], answer, {'question': question})
            chat.append(StepInfo(step, self.max_iterations))
            if isinstance(output, Observation) and output.reflection_needed():
                observation = output
                reflection_string = reflect(self.model, question, observation, what_have_we_learned)

                available_tools = self.get_tools(step)
                planning_string = plan_next_action(self.model, question, available_tools, observation, reflection_string)
                chat.append({'role': 'user', 'content': planning_string})
        return None

    def get_available_tools(self, step: int)-> str:
        tools = self.get_tools(step)
        schemas = get_tool_defs(tools)
        available_tools = self.format_tool_docstrings(schemas)
        return available_tools


@dataclass
class QAProcessorDeep:
    main_processor_config: dict[str, Any]
    sub_processor_config: dict[str, Any]

    def __post_init__(self):
        sub_processor =  QAProcessor(**self.sub_processor_config)

        def delegate(sub_question: str):
            """Delegate a question to a wikipedia expert"""
            logger.info(f"Delegate a question to a wikipedia expert: '{sub_question}'")
            return sub_processor.process(sub_question)

        self.main_processor_config['toolbox'] = [delegate]
        self.main_processor = QAProcessor(**self.main_processor_config)

    def process(self, question: str):
        return self.main_processor.process(question)
