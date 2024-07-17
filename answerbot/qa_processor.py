from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Iterable, runtime_checkable, Protocol

from llm_easy_tools import LLMFunction, get_tool_defs

import logging

from answerbot.chat import Chat, SystemPrompt, Prompt, SystemPrompt
from answerbot.tools.observation import Observation
from answerbot.reflection_result import ReflectionResult 
from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece

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

@dataclass(frozen=True)
class PlanningPrompt(Prompt):
    question: str
    available_tools: list[dict]
    observations: list[Observation] = field(default_factory=list)
    reflection: Optional[str] = None

@dataclass(frozen=True)
class PlanningSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionSystemPrompt(Prompt):
    pass

@dataclass(frozen=True)
class ReflectionPrompt(Prompt):
    memory: KnowledgeBase
    question: str
    observations: list[Observation]


@dataclass(frozen=True)
class QAProcessor:
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
        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            templates=self.prompt_templates,
            templates_dirs=self.prompt_templates_dirs,
            fail_on_tool_error=self.fail_on_tool_error,
            system_prompt=system_prompt,
        )
        return chat

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = self.make_chat(system_prompt=SystemPrompt())
        chat.append(Question(question, self.max_iterations))

        metadata = self.mk_metadata()

        what_have_we_learned = KnowledgeBase()

        observations = []
        reflection_string = None
        for step in range(self.max_iterations + 1):
            planning_string = self.plan_next_action(question, observations, reflection_string)
            chat.append(planning_string)
            tools = self.get_tools(step)
            chat(StepInfo(step, self.max_iterations), tools=tools, metadata=metadata)
            results = chat.process()
            reflection_string = None
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    answer = output
                    logger.info(f"Answer: '{answer}' for question: '{question}'")
                    full_answer = chat.renderer.render_prompt(answer, question=question)
                    return full_answer
                observations.append(output)  # Add new observation to the list
                if isinstance(output, Observation):
                    if output.quotable:
                        reflection_string = self.reflect(question, observations, what_have_we_learned)
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

    def reflect(self, question: str, observations: list[Observation], knowledge_base: KnowledgeBase) -> str:
        chat = self.make_chat(system_prompt=ReflectionSystemPrompt())
        chat(
            ReflectionPrompt(memory=knowledge_base, question=question, observations=observations),
            metadata=self.mk_metadata(['reflection']),
            tools=[ReflectionResult]
        )

        reflections = []
        for reflection in chat.process():
            if observations[-1].source:
                reflections.append(reflection.update_knowledge_base(knowledge_base, observations[-1]))
        reflection_string = '\n'.join(reflections)

        return reflection_string

    def plan_next_action(self, question: str, observations: list[Observation], reflection_string: Optional[str] = None) -> str:
        chat = self.make_chat(system_prompt=PlanningSystemPrompt())

        schemas = get_tool_defs(self.get_tools(0))

        planning_prompt = PlanningPrompt(
            question=question,
            available_tools=schemas,
            observations=observations,
            reflection=reflection_string,
        )

        planning_result = chat(planning_prompt, metadata=self.mk_metadata(['planning']))

        planning_string = f"**My Notes**\n{reflection_string}\n\nHmm what I could do next?\n\n{planning_result}"

        return planning_string


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

if __name__ == "__main__":
    from dotenv import load_dotenv
    import litellm
    import sys

    load_dotenv()
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Configure logging for the chat module
    #chat_logger = logging.getLogger('answerbot.chat')
    #chat_logger.setLevel(logging.DEBUG)
    #chat_logger.addHandler(logging.StreamHandler(sys.stdout))

    qa_processor = QAProcessor(
        toolbox=[],
        max_iterations=1,
        model='gpt-3.5-turbo',
        prompt_templates_dirs=['answerbot/templates/common/', 'answerbot/templates/wiki_researcher/'],
        name='test'
    )

    # Create an example Observation
    observation = Observation(
        content="Constantinople was the capital of the Byzantine Empire. The Byzantine Empire lasted from 330 AD to 1453 AD.",
        source="https://en.wikipedia.org/wiki/Byzantine_Empire",
        operation="Initial search",
        quotable=True
    )

    # Example knowledge base
    what_have_we_learned = KnowledgeBase()
    new_knowledge_piece = KnowledgePiece(
        url="https://www.britannica.com/place/Byzantine-Empire",
        quotes=["The Byzantine Empire was also known as the Eastern Roman Empire and was a continuation of the Roman Empire in its eastern provinces."],
        learned="The Byzantine Empire was a continuation of the Roman Empire in its eastern provinces and was also known as the Eastern Roman Empire."
    )

    # Add the new KnowledgePiece to the knowledge base
    what_have_we_learned.add_knowledge_piece(new_knowledge_piece)

    question = 'What is the capital of the Eastern Roman Empire?'

    reflection_string = qa_processor.reflect(question, [observation], what_have_we_learned)

    print()
    print(reflection_string)
    print()
    print(qa_processor.plan_next_action(question, [observation], reflection_string))