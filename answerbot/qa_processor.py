from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Iterable, runtime_checkable, Protocol

from llm_easy_tools import LLMFunction, get_tool_defs

import logging

from prompete.chat import Chat, SystemPrompt, Prompt
from jinja2 import Environment, FileSystemLoader, ChoiceLoader

from answerbot.tools.observation import Observation, KnowledgePiece, History
from answerbot.reflection_result import ReflectionResult
from answerbot.qa_prompts import (
    Question, Answer, StepInfo, PlanningInsert, ReflectionPrompt, 
    PlanningPrompt, ReflectionSystemPrompt, PlanningSystemPrompt,
    PostProcess,
    indent_and_quote
)

# Configure logging for this module
logger = logging.getLogger('qa_processor')

@runtime_checkable
class HasLLMTools(Protocol):
    def get_llm_tools(self) -> Iterable[Callable]:
        pass

def expand_toolbox(toolbox: list[HasLLMTools|LLMFunction|Callable]) -> list[Callable|LLMFunction]:
    tools = []
    for item in toolbox:
        if isinstance(item, HasLLMTools):
            tools.extend(item.get_llm_tools())
        else:
            tools.append(item)
    return tools

@dataclass(frozen=True)
class QAProcessorOld:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[str, str] = field(default_factory=dict)
    prompt_templates_dirs: list[str] = field(default_factory=list)
    name: Optional[str] = None
    fail_on_tool_error: bool = False

    def get_tools(self, step: int) -> list[Callable|LLMFunction]:
        tools = expand_toolbox(self.toolbox)
        if step < self.max_iterations:
            return[Answer, *tools]
        else:
            return [Answer]

    def make_chat(self, system_prompt: Optional[Prompt] = None) -> Chat:
        template_dirs = ['answerbot/templates/common/'] + self.prompt_templates_dirs
        renderer = Environment(
            loader=ChoiceLoader([
                FileSystemLoader(template_dirs),
                ])
            )
        renderer.filters['indent_and_quote'] = indent_and_quote

        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            fail_on_tool_error=self.fail_on_tool_error,
            system_prompt=system_prompt,
            renderer=renderer,
        )
        chat.template_env.filters['indent_and_quote'] = indent_and_quote
        return chat

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = self.make_chat(system_prompt=SystemPrompt())
        chat.append(Question(question, self.max_iterations))

        metadata = self.mk_metadata()
        history = History()
        new_sources = []

        for step in range(self.max_iterations + 1):
            planning_string = self.plan_next_action(question, history, new_sources)
            chat.append(PlanningInsert(planning_string, history.latest_knowledge(), new_sources))
            new_sources = []
            tools = self.get_tools(step)
            chat(StepInfo(step, self.max_iterations), tools=tools, metadata=metadata)
            results = chat.process()
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    answer = output
                    logger.info(f"Answer: '{answer}' for question: '{question}'")
                    full_answer = chat.render_prompt(answer, question=question)
                    return full_answer
                history.add_observation(output)
                if isinstance(output, Observation):
                    if output.quotable:
                        new_sources = self.reflect(question, history)
            logger.info(f"Step: {step} for question: '{question}'")

        return None

    def mk_metadata(self, tags: Optional[list[str]] = None) -> dict:
        metadata_tags = []
        if tags:
            metadata_tags.extend(tags)
        if self.name:
            metadata_tags.append(self.name)
        if metadata_tags:
            metadata = {'tags': metadata_tags}
        else:
            metadata = {}
        return metadata

    def reflect(self, question: str, history: History) -> list[str]:
        reflection_prompt = ReflectionPrompt(history=history, question=question)
        chat = self.make_chat(system_prompt=ReflectionSystemPrompt())
        chat(
            reflection_prompt,
            metadata=self.mk_metadata(['reflection']),
            tools=[ReflectionResult]
        )
        new_sources = []
        for reflection in chat.process():
            reflection.update_history(history)
            new_sources.extend(reflection.new_sources)
        return new_sources

    def plan_next_action(self, question: str, history: History, new_sources: list[str]) -> str:
        chat = self.make_chat(system_prompt=PlanningSystemPrompt())

        schemas = get_tool_defs(self.get_tools(0))

        planning_prompt = PlanningPrompt(
            question=question,
            available_tools=schemas,
            history=history,
            new_sources=new_sources
        )

        return chat(planning_prompt, metadata=self.mk_metadata(['planning']))


@dataclass(frozen=True)
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[str, str] = field(default_factory=dict)
    prompt_templates_dirs: list[str] = field(default_factory=list)
    name: Optional[str] = None
    fail_on_tool_error: bool = False
    answer_type: str = "full"  # Changed from full_answer: bool = True

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        history = History()
        new_sources = []

        for step in range(self.max_iterations + 1):
            chat = self.make_chat(system_prompt=SystemPrompt())
            self.plan_next_action(chat, question, history, new_sources)
            new_sources = []
            tools = self.get_tools(step)
            chat(StepInfo(step, self.max_iterations), tools=tools, metadata=self.mk_metadata([f'Step {step}']))
            results = chat.process()
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    return self.handle_answer(output, question)
                history.add_observation(output)
                if isinstance(output, Observation):
                    if output.quotable:
                        new_sources = self.reflect(question, history)
            logger.info(f"Step: {step} for question: '{question}'")

        return None

    def handle_answer(self, answer: Answer, question: str) -> str:
        logger.info(f"Answer: '{answer}' for question: '{question}'")
        if self.answer_type == "full":
            full_answer = self.make_chat().render_prompt(answer, question=question)
            return full_answer
        elif self.answer_type == "simple":
            return answer.answer
        elif self.answer_type == "postprocess":
            return self.postprocess_answer(answer, question)
        else:
            raise ValueError(f"Invalid answer type: {self.answer_type}")

    def postprocess_answer(self, answer: Answer, question: str) -> str:
        chat = self.make_chat()
        postprocess_prompt = PostProcess(answer.answer, question)
        result = chat(postprocess_prompt)
        if result.startswith('Compressed: '):
            result = result[9:]  # Remove the 'Implicit: ' prefix
        return result

    def plan_next_action(self, chat: Chat, question: str, history: History, new_sources: list[str]) -> str:
        schemas = get_tool_defs(self.get_tools(0))

        planning_prompt = PlanningPrompt(
            question=question,
            available_tools=schemas,
            history=history,
            new_sources=new_sources
        )

        return chat(planning_prompt, metadata=self.mk_metadata(['planning']))

    def get_tools(self, step: int) -> list[Callable|LLMFunction]:
        tools = expand_toolbox(self.toolbox)
        if step < self.max_iterations:
            return[Answer, *tools]
        else:
            return [Answer]

    def make_chat(self, system_prompt: Optional[Prompt] = None) -> Chat:
        template_dirs = ['answerbot/templates/common/'] + self.prompt_templates_dirs
        renderer = Environment(
            loader=ChoiceLoader([
                FileSystemLoader(template_dirs),
                ])
            )
        renderer.filters['indent_and_quote'] = indent_and_quote

        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            fail_on_tool_error=self.fail_on_tool_error,
            system_prompt=system_prompt,
            renderer=renderer,
        )
        return chat

    def mk_metadata(self, tags: Optional[list[str]] = None) -> dict:
        metadata_tags = []
        if tags:
            metadata_tags.extend(tags)
        if self.name:
            metadata_tags.append(self.name)
        if metadata_tags:
            metadata = {'tags': metadata_tags}
        else:
            metadata = {}
        return metadata

    def reflect(self, question: str, history: History) -> list[str]:
        reflection_prompt = ReflectionPrompt(history=history, question=question)
        chat = self.make_chat(system_prompt=ReflectionSystemPrompt())
        chat(
            reflection_prompt,
            metadata=self.mk_metadata(['reflection']),
            tools=[ReflectionResult]
        )
        new_sources = []
        for reflection in chat.process():
            if isinstance(reflection, ReflectionResult):
                reflection.update_history(history)
                new_sources.extend(reflection.new_sources)
        return new_sources


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
                content = sub_processor.process(sub_question)
                observation = Observation(content, source="wikipedia expert", operation=f'delegate("{sub_question}")', quotable=True)
                return observation

            delegate_fun = LLMFunction(delegate, description=self.delegate_description)

            object.__setattr__(self, 'toolbox', [delegate_fun])

if __name__ == "__main__":
    from dotenv import load_dotenv
    import litellm
    import sys

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Configure logging for the chat module
    chat_logger = logging.getLogger('answerbot.chat')
    chat_logger.setLevel(logging.INFO)
    chat_logger.addHandler(logging.StreamHandler(sys.stdout))

    qa_processor = QAProcessor(
        toolbox=[],
        max_iterations=1,
        model='gpt-3.5-turbo',
        prompt_templates_dirs=['answerbot/templates/common/', 'answerbot/templates/wiki_researcher/'],
        name='test'
    )

    history = History()

    # Create an example Observation
    observation1 = Observation(
        content="The Byzantine Empire was also known as the Eastern Roman Empire and was a continuation of the Roman Empire in its eastern provinces.",
        source="https://en.wikipedia.org/wiki/Byzantine_Empire",
        operation="Initial search",
        quotable=True
    )
    history.add_observation(observation1)

    # Example knowledge base
    new_knowledge_piece = KnowledgePiece(
        source=observation1,
        quotes=["The Byzantine Empire was also known as the Eastern Roman Empire and was a continuation of the Roman Empire in its eastern provinces."],
        content="The Byzantine Empire was a continuation of the Roman Empire in its eastern provinces and was also known as the Eastern Roman Empire."
    )

    history.add_knowledge_piece(new_knowledge_piece)

    observation2 = Observation(
        content="Constantinople was the capital of the Byzantine Empire.",
        source="https://en.wikipedia.org/wiki/Byzantine_Empire",
        operation="read more",
        quotable=True
    )
    history.add_observation(observation2)

    question = 'What is the capital of the Eastern Roman Empire?'

    new_sources = qa_processor.reflect(question, history)

    chat = qa_processor.make_chat()
    print(qa_processor.plan_next_action(chat, question, history, new_sources))