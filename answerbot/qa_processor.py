from dataclasses import dataclass, field
from typing import Callable, Any, Optional

from llm_easy_tools import LLMFunction, get_tool_defs

import logging

from answerbot.chat import Chat, HasLLMTools, SystemPrompt, expand_toolbox
from answerbot.tools.observation import Observation
from answerbot.reflection_result import ReflectionResult 
from answerbot.knowledgebase import KnowledgeBase, KnowledgePiece
from answerbot.qa_prompts import Question, Answer, StepInfo, ReflectionPrompt, ReflectionSystemPrompt, PlanningPrompt, PlanningSystemPrompt

# Configure logging for this module
logger = logging.getLogger('qa_processor')

def format_tool_docstrings(schemas: list[dict]) -> str:
    formatted_list = []
    for schema in schemas:
        func_name = schema['function']['name']
        description = schema['function'].get('description', '')

        # Start with function name and description
        doc = f"- **{func_name}**\n\n"
        doc += "\n".join(f"  {line}" for line in description.split('\n')) + "\n\n"

        # Add parameters section if present
        if 'parameters' in schema['function']:
            doc += "  Parameters:\n\n"
            properties = schema['function']['parameters'].get('properties', {})
            for param, details in properties.items():
                param_type = details.get('type', 'Any')
                param_desc = details.get('description', '')
                doc += f"  {param} : {param_type}\n"
                doc += "\n\n".join(f"      {line}" for line in param_desc.split('\n')) + "\n\n"

        formatted_list.append(doc)

    return "\n".join(formatted_list)


@dataclass(frozen=True)
class QAProcessor:
    toolbox: list[HasLLMTools|LLMFunction|Callable]
    max_iterations: int
    model: str
    prompt_templates: dict[str, str] = field(default_factory=dict)
    prompt_templates_dirs: list[str] = field(default_factory=list)
    name: Optional[str] = None
    fail_on_tool_error: bool = False

    def get_tools(self, step: int) -> list[Callable|LLMFunction|HasLLMTools]:
        if step < self.max_iterations:
            return[Answer, *self.toolbox]
        else:
            return [Answer]

    def make_chat(self, tags: Optional[list[str]] = None) -> Chat:
        chat = Chat(
            model=self.model,
            one_tool_per_step=True,
            templates=self.prompt_templates,
            templates_dirs=self.prompt_templates_dirs,
            fail_on_tool_error=self.fail_on_tool_error,
        )
        return chat

    def process(self, question: str):
        logger.info(f'Processing question: {question}')
        chat = self.make_chat()
        chat.append(SystemPrompt())
        chat.append(Question(question, self.max_iterations))

        metadata = self.mk_metadata()

        what_have_we_learned = KnowledgeBase()

        observations = []
        reflection_string = None
        for step in range(self.max_iterations + 1):
            planning_string = self.plan_next_action(question, observations, reflection_string)
            chat.append({'role': 'user', 'content': planning_string})
            chat.append(StepInfo(step, self.max_iterations))
            tools = self.get_tools(step)
            results = chat.process(tools, metadata=metadata)
            reflection_string = None
            if not results:
                logger.warn("No tool call in a tool loop")
            else:
                output = results[0]
                if isinstance(output, Answer):
                    answer = output
                    logger.info(f"Answer: '{answer}' for question: '{question}'")
                    full_answer = chat.renderer.render_prompt(answer, context={'question': question})
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
        chat = self.make_chat()
        chat.append(ReflectionSystemPrompt())
        chat.append(ReflectionPrompt(memory=knowledge_base, question=question, observations=observations))

        reflections = []
        metadata = self.mk_metadata(['reflection'])
        for reflection in chat.process([ReflectionResult], metadata=metadata):
            if observations[-1].source:
                reflections.append(reflection.update_knowledge_base(knowledge_base, observations[-1]))
        reflection_string = '\n'.join(reflections)

        return reflection_string

    def plan_next_action(self, question: str, observations: list[Observation], reflection_string: Optional[str] = None) -> str:
        chat = self.make_chat(['planning'])
        if chat.templates.get('PlanningSystemPrompt'):
            chat.append(PlanningSystemPrompt())

        schemas = get_tool_defs(expand_toolbox(self.get_tools(0)))
        available_tools_str = format_tool_docstrings(schemas)

        planning_prompt = PlanningPrompt(
            question=question,
            available_tools=available_tools_str,
            observations=observations,
            reflection=reflection_string,
        )

        chat.append(planning_prompt)
        metadata = self.mk_metadata(['planning'])
        response = chat.llm_reply(metadata=metadata)
        planning_result = response.choices[0].message.content

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